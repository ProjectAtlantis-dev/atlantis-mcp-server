import test from 'node:test';
import assert from 'node:assert/strict';
import {GroundControls} from '../src/ground-controls.mjs';
const id='12345678-1234-4234-8234-123456789abc',ownerAccountId='22345678-1234-4234-8234-123456789abc';
const surface={origin:{lat:64,lon:-51},minX:-120,minY:-120,stepM:2,rows:121,cols:121,heights:Array(121*121).fill(10),obstacles:[]};
function fixture(){let now=0;const controls=new GroundControls([],()=>now);
  controls.attach({id,ownerAccountId,terrainAssetId:'amv',definitionId:'patria-amv',pose:{x:0,y:0,headingRad:0},surface});
  return {controls,step(){now+=1000/30;controls.step(1/30);}};}
function dispatch(controls,distance=100){const mission=controls.drive_to({id,actor:'bot',ownerAccountId,requestId:'mission-1',destination:{lat:64+distance/6378137*180/Math.PI,lon:-51}});
  controls.mission_surface({id,missionId:mission.id,surface});return mission;}
test('autonomous mission reaches 100m without input refresh and reports stopped arrival',()=>{
  const {controls,step}=fixture();dispatch(controls);
  for(let i=0;i<3000&&controls.observe(id).mission.status!=='completed';i++)step();
  const state=controls.observe(id);assert.equal(state.mission.status,'completed');
  assert.ok(Math.abs(state.position.y-100)<=1.5);assert.ok(state.speedMps<.05);
});
test('manual possession and mission are mutually exclusive; pause/cancel require mission ID',()=>{
  const {controls,step}=fixture();const mission=dispatch(controls);
  assert.throws(()=>controls.claim({id,actor:'human',ownerAccountId}),/cancel mission/);
  controls.mission_control({id,missionId:mission.id,ownerAccountId,action:'pause'});
  for(let i=0;i<30;i++)step();assert.equal(controls.observe(id).position.y,0);
  assert.throws(()=>controls.mission_control({id,missionId:'wrong',ownerAccountId,action:'resume'}),/mission ID/);
  controls.mission_control({id,missionId:mission.id,ownerAccountId,action:'cancel'});
  assert.ok(controls.claim({id,actor:'human',ownerAccountId}).id);
});
test('mission survives saved-state reconstruction, paused without replay or new UUID',()=>{
  const {controls,step}=fixture();const mission=dispatch(controls);for(let i=0;i<100;i++)step();
  const saved=controls.exportState(),restored=new GroundControls(saved);
  assert.deepEqual(restored.observe(id).position,controls.observe(id).position);
  assert.equal(restored.observe(id).mission.id,mission.id);assert.equal(restored.observe(id).mission.status,'paused');
  assert.equal(restored.observe(id).speedMps,0);
});
test('a building blocking the direct line is routed around',()=>{
  const {controls,step}=fixture();const mission=dispatch(controls);
  controls.mission_surface({id,missionId:mission.id,surface:{...surface,obstacles:[{minX:-5,maxX:5,minY:5,maxY:15}]}});
  for(let i=0;i<5000&&controls.observe(id).mission.status!=='completed';i++)step();const state=controls.observe(id);
  assert.equal(state.mission.status,'completed');assert.ok(state.distanceM>100);
});

test('vehicle backs along an escape waypoint without a false destination-progress timeout',()=>{
 const {controls,step}=fixture();dispatch(controls);
 const v=controls.get(id);v.headingRad=Math.PI;
 // Escape route runs north even though the eventual destination is south.
 v.mission.target={x:0,y:-100};v.mission.bestRemainingM=100;
 v.mission.navigation={points:[{x:0,y:110},{x:50,y:110}],index:0,complete:false};
 for(let i=0;i<2100;i++)step();
 const state=controls.observe(id);
 assert.equal(state.mission.status,'running');assert.equal(state.mission.reason,null);
 assert.ok(state.position.y>80);assert.ok(Math.abs(state.position.x)<.01);
 assert.ok(state.speedMps<0);assert.ok(state.mission.remainingM>180);
});
