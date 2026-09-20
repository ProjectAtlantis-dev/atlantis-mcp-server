import {randomUUID} from 'node:crypto';
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createServer} from 'node:net';
import {mkdtemp,rm} from 'node:fs/promises';
import {DatabaseSync} from 'node:sqlite';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {once} from 'node:events';

test('committed Terrain pose survives abrupt child death; restart does not replay control',async()=>{
  const probe=createServer();probe.listen(0,'127.0.0.1');await once(probe,'listening');
  const port=probe.address().port;await new Promise(resolve=>probe.close(resolve));
  const directory=await mkdtemp(join(tmpdir(),'terrain-recovery-'));
  const database=join(directory,'state.sqlite'),token=randomUUID();let child;
  async function start(){
    child=spawn(process.execPath,[new URL('../src/server.mjs',import.meta.url).pathname,
      '--port',String(port),'--token',token,'--database',database],{stdio:['ignore','pipe','pipe']});
    await Promise.race([once(child.stdout,'data'),once(child,'exit').then(()=>{throw Error('Startup failed');})]);
  }
  async function stop(signal){const exited=once(child,'exit');child.kill(signal);await exited;child=null;}
  async function request(action,payload){
    const response=await fetch(`http://127.0.0.1:${port}/games/recovery/${action}`,{
      method:'POST',headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},body:JSON.stringify(payload)});
    return {status:response.status,body:await response.json()};
  }
  const id='12345678-1234-4234-8234-123456789abc',ownerAccountId='22345678-1234-4234-8234-123456789abc';
  try{
    await start();
    assert.equal((await request('vehicle-control',{operation:'attach',id,ownerAccountId,terrainAssetId:'existing-amv',definitionId:'patria-amv',
      pose:{x:0,y:0,headingRad:0},surface:{origin:{lat:64,lon:-51},minX:-50,minY:-50,stepM:100,rows:2,cols:2,heights:[10,10,10,10]}})).status,200);
    const lease=(await request('vehicle-control',{operation:'claim',id,actor:'human',ownerAccountId})).body;
    assert.equal((await request('vehicle-control',{operation:'drive',id,actor:'human',leaseId:lease.id,sequence:1,throttle:1,steering:0,brake:0,durationMs:2000})).status,200);
    assert.equal((await request('reset',{})).status,400);
    await new Promise(resolve=>setTimeout(resolve,250));
    await stop('SIGKILL');
    const db=new DatabaseSync(database,{readOnly:true});
    const saved=JSON.parse(db.prepare('SELECT state_json FROM simulation_room WHERE game_id=?').get('recovery').state_json);
    db.close();
    // Look up the persisted controller field rather than relying on scene events.
    const vehicles=saved.controlledVehicles;
    assert.ok(vehicles[0].position.y>0);
    await start();
    const restored=(await request('vehicle-control',{operation:'observe',id})).body;
    assert.deepEqual(restored.position,vehicles[0].position);
    assert.equal(restored.speedMps,0);assert.equal(restored.controlled,false);
    assert.equal(restored.controlStatus,'restart-paused');
    const stale=await request('vehicle-control',{operation:'drive',id,actor:'human',leaseId:lease.id,sequence:2,throttle:1,steering:0,brake:0,durationMs:500});
    assert.equal(stale.status,409);assert.equal(stale.body.error,'vehicle_command_rejected');assert.match(stale.body.message,/lease/);
  }finally{if(child)await stop('SIGTERM');await rm(directory,{recursive:true,force:true});}
});
