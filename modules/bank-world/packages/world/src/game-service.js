const sqlite3 = require('sqlite3').verbose();
const { v4: uuidv4, v5: uuidv5, validate: uuidValidate } = require('uuid');

const STARTER_ASSET_NAMESPACE = 'f9941849-c59f-49cf-9ee4-fc92e9d87cd7';
const ACTIVE_BANK_ASSET_STATES = new Set(['active', 'stored', 'in_transit', 'installed', 'deployed']);
const VEHICLE_TYPES = {
    coastal_cargo_boat: {
        mode: 'sea',
        fuelCapacityLiters: 2200,
        fuelLitersPerKm: 2.2,
        cargoCapacityTonnes: 20
    }
};

const SEED_NODES = [
    {
        code: 'NUUK',
        name: 'Nuuk',
        kind: 'port',
        lat: 64.1797,
        lon: -51.7414,
        services: ['market', 'warehouse', 'fuel', 'repair', 'bank']
    },
    {
        code: 'SISIMIUT',
        name: 'Sisimiut',
        kind: 'port',
        lat: 66.9395,
        lon: -53.6735,
        services: ['market', 'warehouse', 'fuel', 'repair']
    },
    {
        code: 'ILULISSAT',
        name: 'Ilulissat',
        kind: 'port',
        lat: 69.2198,
        lon: -51.0986,
        services: ['market', 'warehouse', 'fuel', 'repair']
    }
];

const SEED_ROUTES = [
    {
        id: 'sea:NUUK:SISIMIUT',
        from: 'NUUK',
        to: 'SISIMIUT',
        distanceKm: 322,
        durationSeconds: 10 * 60 * 60,
        path: [[-51.7414, 64.1797], [-52.35, 65.05], [-53.1, 66.0], [-53.6735, 66.9395]]
    },
    {
        id: 'sea:SISIMIUT:NUUK',
        from: 'SISIMIUT',
        to: 'NUUK',
        distanceKm: 322,
        durationSeconds: 10 * 60 * 60,
        path: [[-53.6735, 66.9395], [-53.1, 66.0], [-52.35, 65.05], [-51.7414, 64.1797]]
    },
    {
        id: 'sea:SISIMIUT:ILULISSAT',
        from: 'SISIMIUT',
        to: 'ILULISSAT',
        distanceKm: 278,
        durationSeconds: 8 * 60 * 60,
        path: [[-53.6735, 66.9395], [-53.1, 67.7], [-52.2, 68.5], [-51.0986, 69.2198]]
    },
    {
        id: 'sea:ILULISSAT:SISIMIUT',
        from: 'ILULISSAT',
        to: 'SISIMIUT',
        distanceKm: 278,
        durationSeconds: 8 * 60 * 60,
        path: [[-51.0986, 69.2198], [-52.2, 68.5], [-53.1, 67.7], [-53.6735, 66.9395]]
    }
];

const SEED_MARKETS = [
    ['NUUK', 'food_supplies', 14, 10, 600],
    ['NUUK', 'fuel', 9, 7, 1200],
    ['NUUK', 'fish', 22, 18, 180],
    ['SISIMIUT', 'fish', 11, 8, 900],
    ['SISIMIUT', 'fuel', 11, 8, 700],
    ['SISIMIUT', 'food_supplies', 19, 14, 240],
    ['ILULISSAT', 'fish', 13, 9, 650],
    ['ILULISSAT', 'fuel', 13, 10, 500],
    ['ILULISSAT', 'food_supplies', 21, 16, 200]
];

class GameError extends Error {
    constructor(message, code = 'GAME_ERROR', status = 400) {
        super(message);
        this.name = 'GameError';
        this.code = code;
        this.status = status;
    }
}

function requireString(value, name) {
    if (typeof value !== 'string' || !value.trim()) {
        throw new GameError(`${name} is required`, 'INVALID_ARGUMENT');
    }
    return value.trim();
}

function requireUuid(value, name) {
    const normalized = requireString(value, name);
    if (!uuidValidate(normalized)) {
        throw new GameError(`${name} must be a UUID`, 'INVALID_UUID');
    }
    return normalized;
}

function parseJson(value, fallback = {}) {
    try {
        return value ? JSON.parse(value) : fallback;
    } catch (_error) {
        return fallback;
    }
}

function iso(value) {
    const date = value instanceof Date ? value : new Date(value);
    if (Number.isNaN(date.getTime())) throw new GameError('Invalid time', 'INVALID_TIME');
    return date.toISOString();
}

function mapNode(row) {
    return row && {
        code: row.code,
        name: row.name,
        kind: row.kind,
        lat: row.lat,
        lon: row.lon,
        anchorTileId: row.anchor_tile_id,
        services: parseJson(row.services_json, [])
    };
}

function mapRoute(row) {
    return row && {
        id: row.id,
        mode: row.mode,
        fromNodeCode: row.from_node_code,
        toNodeCode: row.to_node_code,
        distanceKm: row.distance_km,
        durationSeconds: row.duration_seconds,
        risk: row.risk,
        path: parseJson(row.path_json, [])
    };
}

function mapPlayer(row) {
    return row && {
        id: row.id,
        externalUserId: row.external_user_id,
        bankAccountId: row.bank_account_id,
        displayName: row.display_name,
        homeNodeCode: row.home_node_code,
        createdAt: row.created_at
    };
}

function mapVehicle(row) {
    return row && {
        assetId: row.asset_id,
        controllerAccountId: row.controller_account_id,
        vehicleType: row.vehicle_type,
        status: row.status,
        currentNodeCode: row.current_node_code,
        activeTransitId: row.active_transit_id,
        fuelLiters: row.fuel_liters,
        fuelCapacityLiters: row.fuel_capacity_liters,
        cargoCapacityTonnes: row.cargo_capacity_tonnes,
        damage: row.damage,
        updatedAt: row.updated_at
    };
}

function mapTransit(row) {
    return row && {
        id: row.id,
        vehicleAssetId: row.vehicle_asset_id,
        controllerAccountId: row.controller_account_id,
        routeId: row.route_id,
        fromNodeCode: row.from_node_code,
        toNodeCode: row.to_node_code,
        status: row.status,
        departedAt: row.departed_at,
        arrivesAt: row.arrives_at,
        arrivedAt: row.arrived_at,
        fuelUsedLiters: row.fuel_used_liters,
        path: parseJson(row.path_json, []),
        idempotencyKey: row.idempotency_key,
        createdAt: row.created_at
    };
}

function quantityInTonnes(asset) {
    if (asset.unit === 'tonnes' || asset.unit === 'tonne' || asset.unit === 't') return asset.quantity;
    if (asset.unit === 'kg') return asset.quantity / 1000;
    throw new GameError(`Cargo unit ${asset.unit} cannot be converted to tonnes`, 'UNSUPPORTED_CARGO_UNIT');
}

function interpolatePath(path, progress) {
    if (!Array.isArray(path) || path.length === 0) return null;
    if (path.length === 1) return { lon: path[0][0], lat: path[0][1] };
    const clamped = Math.max(0, Math.min(1, progress));
    const scaled = clamped * (path.length - 1);
    const index = Math.min(path.length - 2, Math.floor(scaled));
    const local = scaled - index;
    return {
        lon: path[index][0] + (path[index + 1][0] - path[index][0]) * local,
        lat: path[index][1] + (path[index + 1][1] - path[index][1]) * local
    };
}

class GameService {
    constructor({ dbPath = 'greenland_game.db', bank = null, clock = () => new Date() } = {}) {
        this.dbPath = dbPath;
        this.bank = bank;
        this.clock = clock;
        this.db = null;
        this.writeQueue = Promise.resolve();
    }

    async open() {
        if (this.db) return this;
        this.db = await new Promise((resolve, reject) => {
            const db = new sqlite3.Database(this.dbPath, (error) => error ? reject(error) : resolve(db));
        });
        await this._exec('PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;');
        await this._initSchema();
        await this._seedWorld();
        return this;
    }

    async close() {
        if (!this.db) return;
        const db = this.db;
        this.db = null;
        await new Promise((resolve, reject) => db.close((error) => error ? reject(error) : resolve()));
    }

    _exec(sql) {
        return new Promise((resolve, reject) => this.db.exec(sql, (error) => error ? reject(error) : resolve()));
    }

    _run(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.run(sql, params, function onRun(error) {
                if (error) reject(error);
                else resolve({ changes: this.changes, lastID: this.lastID });
            });
        });
    }

    _get(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.get(sql, params, (error, row) => error ? reject(error) : resolve(row));
        });
    }

    _all(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.all(sql, params, (error, rows) => error ? reject(error) : resolve(rows));
        });
    }

    _write(work) {
        const operation = this.writeQueue.then(async () => {
            await this._run('BEGIN IMMEDIATE');
            try {
                const result = await work();
                await this._run('COMMIT');
                return result;
            } catch (error) {
                await this._run('ROLLBACK').catch(() => {});
                throw error;
            }
        });
        this.writeQueue = operation.catch(() => {});
        return operation;
    }

    async _initSchema() {
        await this._exec(`
            CREATE TABLE IF NOT EXISTS game_node (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                anchor_tile_id TEXT,
                services_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS game_route (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                from_node_code TEXT NOT NULL REFERENCES game_node(code),
                to_node_code TEXT NOT NULL REFERENCES game_node(code),
                distance_km REAL NOT NULL CHECK (distance_km > 0),
                duration_seconds INTEGER NOT NULL CHECK (duration_seconds > 0),
                risk REAL NOT NULL DEFAULT 0 CHECK (risk >= 0 AND risk <= 1),
                path_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS game_player (
                id TEXT PRIMARY KEY,
                external_user_id TEXT NOT NULL UNIQUE,
                bank_account_id TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                home_node_code TEXT NOT NULL REFERENCES game_node(code),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS game_vehicle (
                asset_id TEXT PRIMARY KEY,
                controller_account_id TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('docked', 'in_transit', 'disabled')),
                current_node_code TEXT REFERENCES game_node(code),
                active_transit_id TEXT,
                fuel_liters REAL NOT NULL CHECK (fuel_liters >= 0),
                fuel_capacity_liters REAL NOT NULL CHECK (fuel_capacity_liters > 0),
                cargo_capacity_tonnes REAL NOT NULL CHECK (cargo_capacity_tonnes > 0),
                damage REAL NOT NULL DEFAULT 0 CHECK (damage >= 0 AND damage <= 1),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS game_transit (
                id TEXT PRIMARY KEY,
                vehicle_asset_id TEXT NOT NULL REFERENCES game_vehicle(asset_id),
                controller_account_id TEXT NOT NULL,
                route_id TEXT NOT NULL REFERENCES game_route(id),
                from_node_code TEXT NOT NULL REFERENCES game_node(code),
                to_node_code TEXT NOT NULL REFERENCES game_node(code),
                status TEXT NOT NULL CHECK (status IN ('in_transit', 'arrived', 'cancelled')),
                departed_at TEXT NOT NULL,
                arrives_at TEXT NOT NULL,
                arrived_at TEXT,
                fuel_used_liters REAL NOT NULL,
                path_json TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS game_cargo (
                resource_asset_id TEXT PRIMARY KEY,
                vehicle_asset_id TEXT NOT NULL REFERENCES game_vehicle(asset_id),
                owner_account_id TEXT NOT NULL,
                quantity_tonnes REAL NOT NULL CHECK (quantity_tonnes > 0),
                loaded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS game_market (
                node_code TEXT NOT NULL REFERENCES game_node(code),
                asset_type TEXT NOT NULL,
                buy_price REAL NOT NULL CHECK (buy_price >= 0),
                sell_price REAL NOT NULL CHECK (sell_price >= 0),
                stock REAL NOT NULL CHECK (stock >= 0),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (node_code, asset_type)
            );

            CREATE TABLE IF NOT EXISTS game_event (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                actor_account_id TEXT,
                entity_id TEXT,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS game_vehicle_controller_idx ON game_vehicle(controller_account_id);
            CREATE INDEX IF NOT EXISTS game_transit_active_idx ON game_transit(status, arrives_at);
            CREATE INDEX IF NOT EXISTS game_event_seq_idx ON game_event(seq);
        `);
    }

    async _seedWorld() {
        await this._write(async () => {
            for (const node of SEED_NODES) {
                await this._run(`
                    INSERT OR IGNORE INTO game_node
                    (code, name, kind, lat, lon, services_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                `, [node.code, node.name, node.kind, node.lat, node.lon, JSON.stringify(node.services)]);
            }
            for (const route of SEED_ROUTES) {
                await this._run(`
                    INSERT OR IGNORE INTO game_route
                    (id, mode, from_node_code, to_node_code, distance_km, duration_seconds, risk, path_json)
                    VALUES (?, 'sea', ?, ?, ?, ?, 0.08, ?)
                `, [route.id, route.from, route.to, route.distanceKm, route.durationSeconds, JSON.stringify(route.path)]);
            }
            for (const market of SEED_MARKETS) {
                await this._run(`
                    INSERT OR IGNORE INTO game_market
                    (node_code, asset_type, buy_price, sell_price, stock)
                    VALUES (?, ?, ?, ?, ?)
                `, market);
            }
        });
    }

    async _event(eventType, actorAccountId, entityId, details, createdAt = this.clock()) {
        await this._run(`
            INSERT INTO game_event (event_type, actor_account_id, entity_id, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)
        `, [eventType, actorAccountId || null, entityId || null, JSON.stringify(details || {}), iso(createdAt)]);
    }

    async ensurePlayer({ externalUserId, displayName, bankAccountId = null, homeNodeCode = 'NUUK' }) {
        externalUserId = requireString(externalUserId, 'externalUserId');
        displayName = requireString(displayName, 'displayName');
        const existing = await this._get('SELECT * FROM game_player WHERE external_user_id = ?', [externalUserId]);
        if (existing) return mapPlayer(existing);

        let accountId = bankAccountId;
        if (!accountId) {
            if (!this.bank) throw new GameError('Bank integration is required to resolve a player', 'BANK_REQUIRED', 503);
            const account = await this.bank.resolveAccount({ externalUserId, displayName, accountType: 'player' });
            accountId = account.id;
        }
        requireUuid(accountId, 'bankAccountId');
        const node = await this._get('SELECT code FROM game_node WHERE code = ?', [homeNodeCode]);
        if (!node) throw new GameError('Unknown home node', 'NODE_NOT_FOUND', 404);

        return this._write(async () => {
            const concurrent = await this._get('SELECT * FROM game_player WHERE external_user_id = ?', [externalUserId]);
            if (concurrent) return mapPlayer(concurrent);
            const id = uuidv4();
            await this._run(`
                INSERT INTO game_player (id, external_user_id, bank_account_id, display_name, home_node_code)
                VALUES (?, ?, ?, ?, ?)
            `, [id, externalUserId, accountId, displayName, homeNodeCode]);
            await this._event('player_joined', accountId, id, { externalUserId, homeNodeCode });
            return mapPlayer(await this._get('SELECT * FROM game_player WHERE id = ?', [id]));
        });
    }

    async bootstrapPlayer(identity) {
        const player = await this.ensurePlayer(identity);
        const existing = await this._get(
            'SELECT * FROM game_vehicle WHERE controller_account_id = ? ORDER BY updated_at LIMIT 1',
            [player.bankAccountId]
        );
        if (existing) return { player, vehicle: mapVehicle(existing), created: false };
        if (!this.bank) throw new GameError('Bank integration is required to issue a starter boat', 'BANK_REQUIRED', 503);

        const assetId = uuidv5(`starter-boat:${player.externalUserId}`, STARTER_ASSET_NAMESPACE);
        let asset = await this.bank.getAsset(assetId).catch(() => null);
        if (!asset) {
            asset = await this.bank.issueAsset({
                id: assetId,
                kind: 'vehicle',
                assetType: 'coastal_cargo_boat',
                ownerAccountId: player.bankAccountId,
                metadata: { source: 'greenland-game', starter: true }
            });
        }
        if (asset.ownerAccountId !== player.bankAccountId || asset.kind !== 'vehicle') {
            throw new GameError('Starter vehicle UUID conflicts with bank ownership', 'STARTER_ASSET_CONFLICT', 409);
        }
        const vehicle = await this.registerVehicle({
            assetId,
            controllerAccountId: player.bankAccountId,
            vehicleType: 'coastal_cargo_boat',
            nodeCode: player.homeNodeCode
        });
        return { player, vehicle, created: true };
    }

    async _verifyBankAsset(assetId, ownerAccountId, kind) {
        if (!this.bank) return null;
        const asset = await this.bank.getAsset(assetId);
        if (!asset) throw new GameError('Bank asset not found', 'BANK_ASSET_NOT_FOUND', 404);
        if (asset.ownerAccountId !== ownerAccountId) {
            throw new GameError('Bank asset is not owned by this account', 'ASSET_NOT_OWNED', 403);
        }
        if (asset.kind !== kind) throw new GameError(`Bank asset must be ${kind}`, 'WRONG_ASSET_KIND');
        if (!ACTIVE_BANK_ASSET_STATES.has(asset.status)) {
            throw new GameError('Bank asset is not active', 'ASSET_NOT_ACTIVE', 409);
        }
        return asset;
    }

    async registerVehicle({ assetId, controllerAccountId, vehicleType, nodeCode = 'NUUK' }) {
        requireUuid(assetId, 'assetId');
        requireUuid(controllerAccountId, 'controllerAccountId');
        vehicleType = requireString(vehicleType, 'vehicleType');
        const spec = VEHICLE_TYPES[vehicleType];
        if (!spec) throw new GameError('Unsupported vehicle type', 'VEHICLE_TYPE_NOT_FOUND', 404);
        const node = await this._get('SELECT code FROM game_node WHERE code = ?', [nodeCode]);
        if (!node) throw new GameError('Unknown node', 'NODE_NOT_FOUND', 404);
        await this._verifyBankAsset(assetId, controllerAccountId, 'vehicle');

        return this._write(async () => {
            const existing = await this._get('SELECT * FROM game_vehicle WHERE asset_id = ?', [assetId]);
            if (existing) {
                if (existing.controller_account_id !== controllerAccountId) {
                    throw new GameError('Vehicle already registered to another controller', 'VEHICLE_CONFLICT', 409);
                }
                return mapVehicle(existing);
            }
            await this._run(`
                INSERT INTO game_vehicle
                (asset_id, controller_account_id, vehicle_type, status, current_node_code,
                 fuel_liters, fuel_capacity_liters, cargo_capacity_tonnes)
                VALUES (?, ?, ?, 'docked', ?, ?, ?, ?)
            `, [
                assetId,
                controllerAccountId,
                vehicleType,
                nodeCode,
                spec.fuelCapacityLiters,
                spec.fuelCapacityLiters,
                spec.cargoCapacityTonnes
            ]);
            await this._event('vehicle_registered', controllerAccountId, assetId, { vehicleType, nodeCode });
            return mapVehicle(await this._get('SELECT * FROM game_vehicle WHERE asset_id = ?', [assetId]));
        });
    }

    async _playerFor(externalUserId) {
        const player = await this._get('SELECT * FROM game_player WHERE external_user_id = ?', [externalUserId]);
        if (!player) throw new GameError('Player profile not found', 'PLAYER_NOT_FOUND', 404);
        return mapPlayer(player);
    }

    async getVehicle(assetId) {
        requireUuid(assetId, 'assetId');
        const row = await this._get('SELECT * FROM game_vehicle WHERE asset_id = ?', [assetId]);
        if (!row) throw new GameError('Vehicle not found', 'VEHICLE_NOT_FOUND', 404);
        const cargo = await this._all('SELECT * FROM game_cargo WHERE vehicle_asset_id = ? ORDER BY loaded_at', [assetId]);
        return {
            ...mapVehicle(row),
            cargo: cargo.map((item) => ({
                resourceAssetId: item.resource_asset_id,
                ownerAccountId: item.owner_account_id,
                quantityTonnes: item.quantity_tonnes,
                loadedAt: item.loaded_at
            })),
            cargoLoadTonnes: cargo.reduce((sum, item) => sum + item.quantity_tonnes, 0)
        };
    }

    async getPlayerState(externalUserId, now = this.clock()) {
        await this.advanceWorld({ now });
        const player = await this._playerFor(requireString(externalUserId, 'externalUserId'));
        const vehicles = await this._all(
            'SELECT * FROM game_vehicle WHERE controller_account_id = ? ORDER BY updated_at',
            [player.bankAccountId]
        );
        return {
            player,
            vehicles: await Promise.all(vehicles.map((vehicle) => this.getVehicle(vehicle.asset_id)))
        };
    }

    async quoteVoyage({ externalUserId, vehicleAssetId, destinationNodeCode, now = this.clock() }) {
        const player = await this._playerFor(requireString(externalUserId, 'externalUserId'));
        const vehicle = await this.getVehicle(vehicleAssetId);
        if (vehicle.controllerAccountId !== player.bankAccountId) {
            throw new GameError('Player does not control this vehicle', 'VEHICLE_NOT_CONTROLLED', 403);
        }
        await this._verifyBankAsset(vehicle.assetId, player.bankAccountId, 'vehicle');
        if (vehicle.status !== 'docked' || !vehicle.currentNodeCode) {
            throw new GameError('Vehicle must be docked to depart', 'VEHICLE_NOT_DOCKED', 409);
        }
        destinationNodeCode = requireString(destinationNodeCode, 'destinationNodeCode').toUpperCase();
        const routeRow = await this._get(`
            SELECT * FROM game_route
            WHERE from_node_code = ? AND to_node_code = ? AND mode = ?
        `, [vehicle.currentNodeCode, destinationNodeCode, VEHICLE_TYPES[vehicle.vehicleType].mode]);
        if (!routeRow) throw new GameError('No compatible direct route', 'ROUTE_NOT_FOUND', 404);
        const route = mapRoute(routeRow);
        const loadRatio = Math.min(1, vehicle.cargoLoadTonnes / vehicle.cargoCapacityTonnes);
        const fuelUsedLiters = Number((
            route.distanceKm * VEHICLE_TYPES[vehicle.vehicleType].fuelLitersPerKm * (1 + loadRatio * 0.25)
        ).toFixed(3));
        if (vehicle.fuelLiters < fuelUsedLiters) {
            throw new GameError('Insufficient fuel for route', 'INSUFFICIENT_FUEL', 409);
        }
        const departedAt = new Date(now);
        const arrivesAt = new Date(departedAt.getTime() + route.durationSeconds * 1000);
        return {
            vehicleAssetId: vehicle.assetId,
            route,
            cargoLoadTonnes: vehicle.cargoLoadTonnes,
            fuelUsedLiters,
            fuelRemainingLiters: Number((vehicle.fuelLiters - fuelUsedLiters).toFixed(3)),
            departedAt: departedAt.toISOString(),
            arrivesAt: arrivesAt.toISOString()
        };
    }

    async departVoyage(args) {
        const idempotencyKey = requireString(args.idempotencyKey, 'idempotencyKey');
        const existing = await this._get('SELECT * FROM game_transit WHERE idempotency_key = ?', [idempotencyKey]);
        if (existing) return this.getTransit(existing.id, args.now);
        const quote = await this.quoteVoyage(args);
        const player = await this._playerFor(args.externalUserId);

        return this._write(async () => {
            const duplicate = await this._get('SELECT * FROM game_transit WHERE idempotency_key = ?', [idempotencyKey]);
            if (duplicate) return this._transitProjection(mapTransit(duplicate), args.now || this.clock());
            const current = await this._get('SELECT * FROM game_vehicle WHERE asset_id = ?', [quote.vehicleAssetId]);
            if (!current || current.status !== 'docked' || current.current_node_code !== quote.route.fromNodeCode) {
                throw new GameError('Vehicle state changed before departure', 'VEHICLE_STATE_CHANGED', 409);
            }
            const id = uuidv4();
            await this._run(`
                INSERT INTO game_transit
                (id, vehicle_asset_id, controller_account_id, route_id, from_node_code, to_node_code,
                 status, departed_at, arrives_at, fuel_used_liters, path_json, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, 'in_transit', ?, ?, ?, ?, ?)
            `, [
                id,
                quote.vehicleAssetId,
                player.bankAccountId,
                quote.route.id,
                quote.route.fromNodeCode,
                quote.route.toNodeCode,
                quote.departedAt,
                quote.arrivesAt,
                quote.fuelUsedLiters,
                JSON.stringify(quote.route.path),
                idempotencyKey
            ]);
            await this._run(`
                UPDATE game_vehicle
                SET status = 'in_transit', current_node_code = NULL, active_transit_id = ?,
                    fuel_liters = fuel_liters - ?, updated_at = ?
                WHERE asset_id = ?
            `, [id, quote.fuelUsedLiters, quote.departedAt, quote.vehicleAssetId]);
            await this._event('voyage_departed', player.bankAccountId, id, {
                vehicleAssetId: quote.vehicleAssetId,
                routeId: quote.route.id,
                fromNodeCode: quote.route.fromNodeCode,
                toNodeCode: quote.route.toNodeCode,
                arrivesAt: quote.arrivesAt
            }, quote.departedAt);
            return this._transitProjection(mapTransit(await this._get('SELECT * FROM game_transit WHERE id = ?', [id])), args.now || this.clock());
        });
    }

    _transitProjection(transit, now = this.clock()) {
        const start = Date.parse(transit.departedAt);
        const end = Date.parse(transit.arrivesAt);
        const progress = transit.status === 'arrived'
            ? 1
            : Math.max(0, Math.min(1, (new Date(now).getTime() - start) / (end - start)));
        return { ...transit, progress, position: interpolatePath(transit.path, progress) };
    }

    async getTransit(transitId, now = this.clock()) {
        requireUuid(transitId, 'transitId');
        await this.advanceWorld({ now });
        const row = await this._get('SELECT * FROM game_transit WHERE id = ?', [transitId]);
        if (!row) throw new GameError('Transit not found', 'TRANSIT_NOT_FOUND', 404);
        return this._transitProjection(mapTransit(row), now);
    }

    async advanceWorld({ now = this.clock() } = {}) {
        const nowIso = iso(now);
        return this._write(async () => {
            const due = await this._all(`
                SELECT * FROM game_transit
                WHERE status = 'in_transit' AND arrives_at <= ?
                ORDER BY arrives_at, id
            `, [nowIso]);
            for (const row of due) {
                await this._run(`
                    UPDATE game_transit SET status = 'arrived', arrived_at = ?
                    WHERE id = ? AND status = 'in_transit'
                `, [nowIso, row.id]);
                await this._run(`
                    UPDATE game_vehicle
                    SET status = 'docked', current_node_code = ?, active_transit_id = NULL, updated_at = ?
                    WHERE asset_id = ? AND active_transit_id = ?
                `, [row.to_node_code, nowIso, row.vehicle_asset_id, row.id]);
                await this._event('voyage_arrived', row.controller_account_id, row.id, {
                    vehicleAssetId: row.vehicle_asset_id,
                    nodeCode: row.to_node_code
                }, now);
            }
            return { advancedAt: nowIso, arrivals: due.length };
        });
    }

    async loadCargo({ externalUserId, vehicleAssetId, resourceAssetId, now = this.clock() }) {
        if (!this.bank) throw new GameError('Bank integration is required to verify cargo', 'BANK_REQUIRED', 503);
        const player = await this._playerFor(requireString(externalUserId, 'externalUserId'));
        const vehicle = await this.getVehicle(vehicleAssetId);
        if (vehicle.controllerAccountId !== player.bankAccountId) {
            throw new GameError('Player does not control this vehicle', 'VEHICLE_NOT_CONTROLLED', 403);
        }
        if (vehicle.status !== 'docked') throw new GameError('Vehicle must be docked to load', 'VEHICLE_NOT_DOCKED', 409);
        const asset = await this._verifyBankAsset(requireUuid(resourceAssetId, 'resourceAssetId'), player.bankAccountId, 'resource_lot');
        const quantityTonnes = quantityInTonnes(asset);
        if (vehicle.cargoLoadTonnes + quantityTonnes > vehicle.cargoCapacityTonnes + 1e-9) {
            throw new GameError('Cargo capacity exceeded', 'CARGO_CAPACITY_EXCEEDED', 409);
        }
        return this._write(async () => {
            const existing = await this._get('SELECT * FROM game_cargo WHERE resource_asset_id = ?', [resourceAssetId]);
            if (existing) {
                if (existing.vehicle_asset_id === vehicleAssetId) return this.getVehicle(vehicleAssetId);
                throw new GameError('Resource lot is already loaded elsewhere', 'CARGO_ALREADY_LOADED', 409);
            }
            await this._run(`
                INSERT INTO game_cargo
                (resource_asset_id, vehicle_asset_id, owner_account_id, quantity_tonnes, loaded_at)
                VALUES (?, ?, ?, ?, ?)
            `, [resourceAssetId, vehicleAssetId, player.bankAccountId, quantityTonnes, iso(now)]);
            await this._event('cargo_loaded', player.bankAccountId, resourceAssetId, { vehicleAssetId, quantityTonnes }, now);
            return this.getVehicle(vehicleAssetId);
        });
    }

    async unloadCargo({ externalUserId, vehicleAssetId, resourceAssetId, now = this.clock() }) {
        const player = await this._playerFor(requireString(externalUserId, 'externalUserId'));
        const vehicle = await this.getVehicle(vehicleAssetId);
        if (vehicle.controllerAccountId !== player.bankAccountId) {
            throw new GameError('Player does not control this vehicle', 'VEHICLE_NOT_CONTROLLED', 403);
        }
        if (vehicle.status !== 'docked') throw new GameError('Vehicle must be docked to unload', 'VEHICLE_NOT_DOCKED', 409);
        return this._write(async () => {
            const result = await this._run(`
                DELETE FROM game_cargo
                WHERE resource_asset_id = ? AND vehicle_asset_id = ? AND owner_account_id = ?
            `, [resourceAssetId, vehicleAssetId, player.bankAccountId]);
            if (!result.changes) throw new GameError('Cargo lot is not loaded on this vehicle', 'CARGO_NOT_FOUND', 404);
            await this._event('cargo_unloaded', player.bankAccountId, resourceAssetId, {
                vehicleAssetId,
                nodeCode: vehicle.currentNodeCode
            }, now);
            return this.getVehicle(vehicleAssetId);
        });
    }

    async getWorldState({ now = this.clock(), afterEventSeq = 0 } = {}) {
        await this.advanceWorld({ now });
        const [nodes, routes, markets, vehicles, transits, events] = await Promise.all([
            this._all('SELECT * FROM game_node ORDER BY code'),
            this._all('SELECT * FROM game_route ORDER BY id'),
            this._all('SELECT * FROM game_market ORDER BY node_code, asset_type'),
            this._all('SELECT * FROM game_vehicle ORDER BY asset_id'),
            this._all("SELECT * FROM game_transit WHERE status = 'in_transit' ORDER BY arrives_at"),
            this._all('SELECT * FROM game_event WHERE seq > ? ORDER BY seq LIMIT 500', [Number(afterEventSeq) || 0])
        ]);
        const nodeByCode = new Map(nodes.map((node) => [node.code, node]));
        return {
            serverTime: iso(now),
            renderer: 'webgpu',
            nodes: nodes.map(mapNode),
            routes: routes.map(mapRoute),
            markets: markets.map((row) => ({
                nodeCode: row.node_code,
                assetType: row.asset_type,
                buyPrice: row.buy_price,
                sellPrice: row.sell_price,
                stock: row.stock,
                updatedAt: row.updated_at
            })),
            vehicles: vehicles.map((row) => {
                const vehicle = mapVehicle(row);
                if (vehicle.status === 'docked') {
                    const node = nodeByCode.get(vehicle.currentNodeCode);
                    return { ...vehicle, position: node ? { lat: node.lat, lon: node.lon } : null };
                }
                const transit = transits.find((candidate) => candidate.id === vehicle.activeTransitId);
                return {
                    ...vehicle,
                    position: transit ? this._transitProjection(mapTransit(transit), now).position : null
                };
            }),
            activeTransits: transits.map((row) => this._transitProjection(mapTransit(row), now)),
            events: events.map((row) => ({
                seq: row.seq,
                eventType: row.event_type,
                actorAccountId: row.actor_account_id,
                entityId: row.entity_id,
                details: parseJson(row.details_json),
                createdAt: row.created_at
            })),
            nextEventSeq: events.length ? events[events.length - 1].seq : Number(afterEventSeq) || 0
        };
    }
}

module.exports = {
    GameError,
    GameService,
    SEED_NODES,
    SEED_ROUTES,
    VEHICLE_TYPES,
    interpolatePath
};
