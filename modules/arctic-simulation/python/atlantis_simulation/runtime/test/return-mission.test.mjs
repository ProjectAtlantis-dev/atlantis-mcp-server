import test from 'node:test';
import assert from 'node:assert/strict';
import {GroundControls} from '../src/ground-controls.mjs';
const id='12345678-1234-4234-8234-123456789abc',ownerAccountId='22345678-1234-4234-8234-123456789abc';
const surface={origin:{lat:64,lon:-51},minX:-120,minY:-120,stepM:2,rows:121,cols:121,heights:Array(14641).fill(10),obstacles:[]};
const origin={lat:64,lon:-51},destination={lat:64+30/6378137*180/Math.PI,lon:-51};
function fixture(flight=false){let now=0;let c=new GroundControls([],()=>now);
 c.attach({id,ownerAccountId,terrainAssetId:'test',definitionId:flight?'black-hornet':'patria-amv',pose:{x:0,y:0,headingRad:0},sourcePose:{z:10},surface});
 const parameters={id,ownerAccountId,actor:'lobster',requestId:'roundtrip',destination,
  returnDestination:flight?{...origin,altitudeM:10.3,landing:true}:origin,...(flight?{altitudeM:40,landing:false}:{})};
 return {get c(){return c;},parameters,dispatch(extra={}){return c[flight?'fly_to':'drive_to']({...parameters,...extra});},
  restore(){c=new GroundControls(c.exportState(),()=>now);},
  step(){const m=c.observe(id).mission;if(m?.status==='queued')c.mission_surface({id,missionId:m.id,surface});now+=1000/30;c.step(1/30);},
  until(predicate){for(let n=0;n<9000;n++){this.step();if(predicate(c.observe(id)))return c.observe(id);}assert.fail(JSON.stringify(c.observe(id).mission));}};
}
for(const flight of [false,true])test(`${flight?'aircraft':'ground'} coordinates dispatch outbound and exactly one durable return`,()=>{
 const f=fixture(flight),outbound=f.dispatch();
 const returning=f.until(v=>v.mission.leg==='return');
 assert.equal(returning.mission.parentMissionId,outbound.id);
 assert.deepEqual(returning.mission.destination,origin);
 assert.equal(f.c.mission_status({id}).history[0].status,'completed');
 assert.equal(f.dispatch().returnMissionId,returning.mission.id);
 f.restore();assert.equal(f.c.observe(id).mission.status,'paused');
 f.c.mission_control({id,ownerAccountId,missionId:returning.mission.id,action:'resume'});
 const final=f.until(v=>v.mission.status==='completed');
 assert.ok(Math.hypot(final.position.x,final.position.y)<=1.5);
 if(flight){assert.equal(final.mission.reason,'landed');assert.ok(Math.abs(final.position.z-10.3)<.2);}
 for(let i=0;i<60;i++)f.step();
 assert.equal(f.c.mission_status({id}).history.length,1);
 assert.equal(f.c.observe(id).mission.id,returning.mission.id);
});
test('task waits for explicit completion, survives restart and pause, completion retries do not create another return',()=>{
 const f=fixture(),m=f.dispatch({waitForTask:true});
 assert.throws(()=>f.c.mission_control({id,ownerAccountId,missionId:m.id,action:'complete_task'}),/awaiting_task/);
 f.until(v=>v.mission.status==='awaiting_task');
 const p=f.c.observe(id).position;f.restore();
 for(let i=0;i<60;i++)f.step();assert.deepEqual(f.c.observe(id).position,p);
 f.c.mission_control({id,ownerAccountId,missionId:m.id,action:'pause'});
 f.c.mission_control({id,ownerAccountId,missionId:m.id,action:'resume'});
 assert.equal(f.c.observe(id).mission.status,'awaiting_task');
 const action={id,ownerAccountId,missionId:m.id,action:'complete_task'};
 assert.throws(()=>f.c.mission_control({...action,ownerAccountId:id}),/ownership/);
 assert.equal(f.c.mission_control(action).status,'completed');
 const ret=f.c.observe(id).mission;assert.equal(ret.leg,'return');
 assert.equal(f.c.mission_control(action).id,m.id);
 assert.equal(f.c.observe(id).mission.id,ret.id);
});
test('blocked or cancelled outbound never queues return; invalid or changed return terms reject',()=>{
 const f=fixture();assert.throws(()=>f.dispatch({returnDestination:{lat:NaN,lon:0}}),/invalid return/);
 const m=f.dispatch();assert.throws(()=>f.dispatch({returnDestination:destination}),/terms conflict/);
 assert.throws(()=>f.dispatch({waitForTask:true}),/terms conflict/);
 f.c.mission_surface({id,missionId:m.id,error:'no surveyed terrain'});
 for(let i=0;i<60;i++)f.step();assert.equal(f.c.observe(id).mission.status,'blocked');
 f.c.mission_control({id,ownerAccountId,missionId:m.id,action:'cancel'});
 for(let i=0;i<60;i++)f.step();assert.equal(f.c.observe(id).mission.status,'cancelled');
 assert.equal(f.c.mission_status({id}).history.length,0);
});
