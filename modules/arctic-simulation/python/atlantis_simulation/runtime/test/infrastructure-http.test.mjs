import {randomUUID} from 'node:crypto';
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createServer} from 'node:net';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {once} from 'node:events';
test('authenticated infrastructure routes persist across isolated server restart',async()=>{
 const probe=createServer();probe.listen(0,'127.0.0.1');await once(probe,'listening');const port=probe.address().port;await new Promise(r=>probe.close(r));
 const dir=await mkdtemp(join(tmpdir(),'atlantis-infra-test-')),token=randomUUID();let child;
 async function start(){
  child=spawn(process.execPath,[new URL('../src/server.mjs',import.meta.url).pathname,'--port',String(port),'--token',token,'--database',join(dir,'test.db')],{stdio:['ignore','pipe','pipe']});
  await Promise.race([once(child.stdout,'data'),once(child,'exit').then(()=>{throw Error('test server exited before startup');})]);
 }
 async function stop(){const exited=once(child,'exit');child.kill('SIGTERM');await exited;child=null;}
 async function request(path,method='GET',body){const response=await fetch(`http://127.0.0.1:${port}/games/test/${path}`,{method,headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});return {status:response.status,body:await response.json()};}
 try{
  await start();assert.equal((await fetch(`http://127.0.0.1:${port}/games/test/infrastructure`)).status,401);
  assert.ok((await request('infrastructure-catalog')).body.models.length>=33);
  assert.equal((await request('infrastructure','POST',{id:'test-pad',modelId:'network-radome',position:{x:12,y:13,z:4}})).status,201);
  assert.equal((await request('infrastructure','PATCH',{id:'test-pad',position:{x:20,y:30,z:4},headingDeg:90})).status,200);
  assert.equal((await request('infrastructure','POST',{id:'test-facility',modelId:'future-city-water-plant',position:{x:40,y:30,z:4}})).status,201);
  assert.equal((await request('component-command','POST',{id:'test-facility',action:'facility_freight_open',expectedRevision:0})).status,200);
  const owner=randomUUID(),policy={version:1,ownerAccountId:owner,allowedAccountIds:[],interactionRadiusM:5,bounds:{minX:-2,maxX:2,minY:-2,maxY:2,minZ:0,maxZ:4}};
  assert.equal((await request('infrastructure-access','POST',{operation:'configure',id:'test-facility',accountId:owner,policy})).status,200);
  assert.equal((await request('component-command','POST',{id:'test-facility',action:'facility_freight_close',expectedRevision:1})).status,400);
  const access=await request('infrastructure-access','POST',{operation:'discover',id:'test-facility',accountId:owner,subjects:[]});assert.equal(access.status,200);assert.deepEqual(access.body,{protected:true,subjects:[]});
  assert.equal((await request('infrastructure-access','POST',{operation:'command',id:'test-facility',action:'facility_freight_close',expectedRevision:1,accountId:owner,subjectKind:'camera',subjectId:randomUUID(),position:{x:40,y:30,z:4}})).status,400);
  await stop();await start();const snapshot=(await request('snapshot')).body;
  assert.deepEqual(snapshot.infrastructure[0].position,{x:20,y:30,z:4});
  const facility=snapshot.infrastructure.find(e=>e.id==='test-facility');assert.equal(facility.componentState.freightTarget,1);assert.equal(facility.componentState.revision,1);
  assert.equal((await request('component-command','POST',{id:'test-facility',action:'facility_entry_inner',expectedRevision:1})).status,400);
  assert.equal((await request('infrastructure','DELETE',{id:'test-pad'})).status,200);
  assert.equal((await request('infrastructure','DELETE',{id:'test-facility'})).status,200);
  assert.deepEqual((await request('infrastructure')).body.entities,[]);
 }finally{if(child)await stop();await rm(dir,{recursive:true,force:true});}
});
