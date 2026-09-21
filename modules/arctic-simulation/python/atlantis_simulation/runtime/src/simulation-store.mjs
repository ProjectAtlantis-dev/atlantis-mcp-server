import {createHash} from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

/** Durable local adapter for simulation checkpoints and append-only run events. */
export class SimulationStore {
  constructor(filename) {
    if (typeof filename !== 'string' || !filename.trim()) throw new Error('simulation database path is required');
    this.filename = resolve(filename);
    mkdirSync(dirname(this.filename), { recursive: true });
    this.database = new DatabaseSync(this.filename);
    this.database.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = FULL;
      PRAGMA foreign_keys = ON;
      CREATE TABLE IF NOT EXISTS simulation_terrain (
        terrain_key TEXT PRIMARY KEY,
        surface_json TEXT NOT NULL
      ) STRICT;
      CREATE TABLE IF NOT EXISTS simulation_room (
        game_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        state_json TEXT NOT NULL,
        last_event_sequence INTEGER NOT NULL,
        updated_at TEXT NOT NULL
      ) STRICT;
      CREATE TABLE IF NOT EXISTS simulation_event (
        game_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        tick INTEGER NOT NULL,
        time_seconds REAL NOT NULL,
        event_type TEXT NOT NULL,
        event_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY (game_id, run_id, sequence)
      ) STRICT;
      CREATE INDEX IF NOT EXISTS simulation_event_game_run
        ON simulation_event (game_id, run_id, sequence);
    `);
    // Surface objects are replaced by attach/mission_surface, never integrated.
    this.surfaceKeys=new WeakMap();
    this.terrainInsert=this.database.prepare('INSERT OR IGNORE INTO simulation_terrain (terrain_key,surface_json) VALUES (?,?)');
    this.terrainSelect=this.database.prepare('SELECT surface_json FROM simulation_terrain WHERE terrain_key=?');
    this.roomUpsert = this.database.prepare(`
      INSERT INTO simulation_room
        (game_id, run_id, state_json, last_event_sequence, updated_at)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(game_id) DO UPDATE SET
        run_id = excluded.run_id,
        state_json = excluded.state_json,
        last_event_sequence = excluded.last_event_sequence,
        updated_at = excluded.updated_at
    `);
    this.eventInsert = this.database.prepare(`
      INSERT OR IGNORE INTO simulation_event
        (game_id, run_id, sequence, tick, time_seconds, event_type, event_json, recorded_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    this.persistedSequence = this.database.prepare(`
      SELECT COALESCE(MAX(sequence), 0) AS sequence
      FROM simulation_event WHERE game_id = ? AND run_id = ?
    `);
    this.roomsSelect = this.database.prepare(`
      SELECT game_id, run_id, state_json, last_event_sequence
      FROM simulation_room ORDER BY game_id
    `);
    this.eventsSelect = this.database.prepare(`
      SELECT event_json FROM (
        SELECT sequence, event_json FROM simulation_event
        WHERE game_id = ? AND run_id = ?
        ORDER BY sequence DESC LIMIT 2000
      ) ORDER BY sequence
    `);
  }

  saveRoom(room) {
    const now = new Date().toISOString();
    const row = this.persistedSequence.get(room.id, room.runId);
    const after = Number(row?.sequence ?? 0);
    const events = room.eventsAfter(after);
    const committedSurfaces=[];
    this.database.exec('BEGIN IMMEDIATE');
    try {
      const encodeSurface=surface=>{
        const cached=this.surfaceKeys.get(surface);
        if(cached)return {terrainRef:cached};
        const json=JSON.stringify(surface),key=createHash('sha256').update(json).digest('hex');
        this.terrainInsert.run(key,json);committedSurfaces.push([surface,key]);
        return {terrainRef:key};
      };
      const state=room.exportState({encodeSurface});state.storageVersion=2;
      for (const event of events) {
        this.eventInsert.run(
          room.id,
          room.runId,
          event.sequence,
          event.tick,
          event.timeSeconds,
          event.type,
          JSON.stringify(event),
          now,
        );
      }
      this.roomUpsert.run(room.id, room.runId, JSON.stringify(state), room.eventSequence, now);
      this.database.exec('COMMIT');
      for(const [surface,key] of committedSurfaces)this.surfaceKeys.set(surface,key);
    } catch (error) {
      this.database.exec('ROLLBACK');
      throw error;
    }
  }

  saveEngine(engine) {
    for (const room of engine.rooms.values()) this.saveRoom(room);
  }

  restoreEngine(engine) {
    let restored = 0;
    for (const row of this.roomsSelect.all()) {
      const state = JSON.parse(row.state_json);
      if(state.storageVersion===2){
        const surfaces=new Map();
        for(const vehicle of state.controlledVehicles){
          const key=vehicle.surface?.terrainRef;
          if(typeof key!=='string')throw Error('Missing durable terrain reference');
          if(!surfaces.has(key)){
            const terrain=this.terrainSelect.get(key);
            if(!terrain)throw Error('Durable terrain grid is missing: '+key);
            if(createHash('sha256').update(terrain.surface_json).digest('hex')!==key)throw Error('Durable terrain grid checksum mismatch');
            surfaces.set(key,JSON.parse(terrain.surface_json));
          }
          vehicle.surface=surfaces.get(key);
        }
        state.storageVersion=1;
      }
      if (state.runId !== row.run_id) throw new Error(`persisted run id mismatch for game: ${row.game_id}`);
      if (state.eventSequence !== row.last_event_sequence) throw new Error(`persisted event sequence mismatch for game: ${row.game_id}`);
      const durableSequence = Number(this.persistedSequence.get(row.game_id, row.run_id)?.sequence ?? 0);
      if (durableSequence !== row.last_event_sequence) throw new Error(`event ledger is incomplete for game: ${row.game_id}`);
      state.events = this.eventsSelect.all(row.game_id, row.run_id).map(eventRow => JSON.parse(eventRow.event_json));
      engine.restore(row.game_id, state);
      restored += 1;
    }
    return restored;
  }

  eventCount(gameId, runId) {
    return Number(this.persistedSequence.get(gameId, runId)?.sequence ?? 0);
  }

  close() {
    this.database.close();
  }
}
