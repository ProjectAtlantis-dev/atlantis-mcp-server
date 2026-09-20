// Explicit opt-in test against a disposable dedicated Redis; never flushes data.
const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');
const { createRedisRealtimeBridgeFromEnv } = require('../packages/realtime/src/create-redis-bridge');
const { createEnvelope } = require('../packages/protocol/src');
const {start}=require('../apps/api/src/server');
const {mkdtemp,rm}=require('node:fs/promises');
const {tmpdir}=require('node:os');
const {join}=require('node:path');

async function main() {
  assert.throws(() => createRedisRealtimeBridgeFromEnv({env:{}}), /Explicit dedicated/);
  const bridge = createRedisRealtimeBridgeFromEnv();
  const room = randomUUID();
  try {
    await bridge.connect();
    let received;
    const delivered = new Promise(resolve => { received = resolve; });
    const unsubscribe = await bridge.bridge.subscribe(room, received);
    const envelope = createEnvelope({kind:'snapshot',roomId:room,sequence:1,tick:30,payload:{assetId:randomUUID()}});
    await bridge.bridge.publish(room,envelope);
    let timeout;
    try {
      assert.deepEqual(await Promise.race([delivered,new Promise((_,reject)=>{timeout=setTimeout(()=>reject(Error('pubsub timeout')),3000);})]),envelope);
    } finally { clearTimeout(timeout); }
    assert.deepEqual((await bridge.bridge.replay(room))[0].envelope,envelope);
    assert.equal(await bridge.bridge.acquireLease({resourceId:room,holderId:'one'}),true);
    assert.equal(await bridge.bridge.acquireLease({resourceId:room,holderId:'two'}),false);
    assert.equal(await bridge.bridge.releaseLease({resourceId:room,holderId:'two'}),false);
    assert.equal(await bridge.bridge.releaseLease({resourceId:room,holderId:'one'}),true);
    await unsubscribe();
    // Reconnect separate clients and prove replay is server state, not client memory.
    await bridge.disconnect();
    const next = createRedisRealtimeBridgeFromEnv();
    try {
      await next.connect();
      assert.deepEqual((await next.bridge.replay(room))[0].envelope,envelope);
      const directory=await mkdtemp(join(tmpdir(),'atlantis-redis-world-'));
      let service;
      try {
        service=await start({...process.env,NODE_ENV:'test',PORT:'0',
          BANK_DB_PATH:join(directory,'bank.test.sqlite'),WORLD_DB_PATH:join(directory,'world.test.sqlite'),
          GREENLAND_WORLD_ROOM_ID:room});
        const player=await service.runtime.world.bootstrapPlayer({externalUserId:'x_user:99',displayName:'test-only'});
        const deadline=Date.now()+3000;
        let projected;
        while(Date.now()<deadline){
          projected=(await next.bridge.replay(room)).find(message=>
            message.envelope.payload.vehicles?.some(v=>v.assetId===player.vehicle.assetId));
          if(projected)break;
          await new Promise(resolve=>setTimeout(resolve,30));
        }
        assert.ok(projected,'server-owned loop did not relay canonical world vehicles');
        assert.equal(projected.envelope.payload.authority,'greenland_world');
        console.log('PASS: autonomous world loop -> Redis replay contains the original bank vehicle UUID');
      } finally {if(service)await service.close();await rm(directory,{recursive:true,force:true});}
    } finally { await next.disconnect(); }
    console.log('PASS: dedicated Redis pubsub, durable replay across reconnect, exclusive lease and owner-checked release');
  } finally {
    if (bridge.publisher.status !== 'end') await bridge.disconnect();
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
