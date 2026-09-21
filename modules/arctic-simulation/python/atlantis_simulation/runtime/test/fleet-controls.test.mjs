import test from 'node:test';
import assert from 'node:assert/strict';
import {GroundControls} from '../src/ground-controls.mjs';
import {planTerrainRoute} from '../src/terrain-route.mjs';
const id='12345678-1234-4234-8234-123456789abc',ownerAccountId='22345678-1234-4234-8234-123456789abc';
const surface={origin:{lat:64,lon:-51},minX:-1500,minY:-1500,stepM:10,rows:301,cols:301,heights:Array(90601).fill(10),water:Array(90601).fill(false)};
const flight={maxSpeedMs:220,stallSpeedMs:32,takeoffSpeedMs:45,accelMs2:18,climbRateMs:25,descendRateMs:20,yawRateRad:.55,rollRateRad:.65};
function fixture(definitionId,definition={},z=10){let now=0;const c=new GroundControls([],()=>now),s=structuredClone(surface);if(definitionId==='patrol-boat'){s.water.fill(true);s.heights.fill(0);s.navigationDomain='water';}c.attach({id,ownerAccountId,definitionId,definition,terrainAssetId:'test-'+definitionId,pose:{x:0,y:0,headingRad:0},sourcePose:{z},surface:s});return {c,s,step(){now+=1000/30;c.step(1/30);}};}
function command(y){return {id,ownerAccountId,actor:'test',requestId:'go',destination:{lat:64+y/6378137*180/Math.PI,lon:-51}};}
test('Hrim drives under its own speed envelope and retains its controller on restart',()=>{
 const {c,s,step}=fixture('at1-hrim'),m=c.drive_to(command(100));c.mission_surface({id,missionId:m.id,surface:s});let max=0;
 for(let i=0;i<6000;i++){step();max=Math.max(max,c.observe(id).speedMps);if(c.observe(id).mission.status==='completed')break;}
 assert.equal(c.observe(id).mission.status,'completed');assert.ok(max>5&&max<=40/3.6);assert.equal(new GroundControls(c.exportState()).observe(id).definitionId,'at1-hrim');
});
test('boat reaches a water waypoint and retains auto-return instructions without driving onto land',()=>{
 const {c,s,step}=fixture('patrol-boat',{boat:{maxSpeedMs:18,accelMs2:4.5,rudderTurnRadS2:.9,yawDamping:1.05}},.5);
 const m=c.sail_to({...command(100),autoReturn:true});c.mission_surface({id,missionId:m.id,surface:s});
 for(let i=0;i<6000;i++){step();if(c.observe(id).mission.leg==='return')break;}
 const v=c.observe(id);assert.equal(v.mission.leg,'return');assert.equal(v.mission.type,'sail_to');assert.equal(v.position.z,.5);assert.equal(v.authority,'server-boat-v1');
 c.mission_surface({id,missionId:v.mission.id,surface:s});for(let i=0;i<12000;i++){step();if(c.observe(id).mission.status==='completed')break;}assert.equal(c.observe(id).mission.status,'completed');assert.ok(Math.hypot(c.observe(id).position.x,c.observe(id).position.y)<1.5);
 const land=structuredClone(s);land.water.fill(false);assert.throws(()=>planTerrainRoute(land,{x:0,y:0},{x:0,y:100}),/traversable/);
 assert.throws(()=>c.drive_to(command(100)),/sail_to/);
});
test('boat route goes around a land peninsula',()=>{
 const s={...structuredClone(surface),navigationDomain:'water'};s.water.fill(true);
 for(let row=150;row<301;row++)for(let col=149;col<=151;col++)s.water[row*301+col]=false;
 const route=planTerrainRoute(s,{x:-100,y:300},{x:100,y:300});assert.ok(route.complete);assert.ok(route.points.some(p=>p.y<0));
});
test('fixed wing rolls, climbs, crosses a waypoint and keeps flying in loiter',()=>{
 const {c,s,step}=fixture('rq180',{flight}),m=c.fly_to({...command(900),altitudeM:100});c.mission_surface({id,missionId:m.id,surface:s});let sawRoll=false;
 for(let i=0;i<6000;i++){step();sawRoll ||=c.observe(id).flightPhase==='takeoff-roll';if(c.observe(id).mission.status==='completed')break;}
 const v=c.observe(id);assert.equal(v.mission.status,'completed');assert.ok(sawRoll);assert.ok(v.speedMps>=flight.stallSpeedMs);assert.equal(v.authority,'server-fixed-wing-v1');
 const before=v.position;for(let i=0;i<60;i++)step();assert.notDeepEqual(c.observe(id).position,before);assert.equal(c.observe(id).flightPhase,'loiter');assert.ok(Math.abs(c.observe(id).rollRad)>0);
});
test('fixed-wing cannot use VTOL landing or ground-departure auto-return',()=>{
 const {c}=fixture('rq180',{flight});assert.throws(()=>c.fly_to({...command(900),altitudeM:100,landing:true}),/landing/);assert.throws(()=>c.fly_to({...command(900),altitudeM:100,autoReturn:true}),/airborne departure/);
 const caps=c.capabilities({id});assert.equal(caps.landing,false);assert.equal(caps.autoReturn,false);assert.equal(caps.takeoffHeading,true);
});
test('fixed-wing rejects an obstructed takeoff roll without moving',()=>{
 const {c,s,step}=fixture('rq180',{flight});s.obstacles=[{minX:-10,maxX:10,minY:20,maxY:40,maxZ:30}];const m=c.fly_to({...command(900),altitudeM:100});c.mission_surface({id,missionId:m.id,surface:s});step();assert.equal(c.observe(id).mission.status,'blocked');assert.match(c.observe(id).mission.reason,/takeoff_heading/);assert.equal(c.observe(id).position.y,0);
});
test('boat brakes on entering the arrival radius instead of circling a close waypoint',()=>{
 const {c,s,step}=fixture('patrol-boat',{boat:{maxSpeedMs:18,accelMs2:4.5,rudderTurnRadS2:.9,yawDamping:1.05}},.5);
 c.get(id).headingRad=335*Math.PI/180;const target={x:12.67855,y:27.18923};
 const p=command(target.y);p.destination.lon+=target.x/(6378137*Math.cos(64*Math.PI/180))*180/Math.PI;
 const m=c.sail_to(p);c.mission_surface({id,missionId:m.id,surface:s});
 for(let i=0;i<3000;i++){step();if(c.observe(id).mission.status==='completed')break;}
 assert.equal(c.observe(id).mission.status,'completed');assert.ok(c.observe(id).distanceM<40);
});
test('fixed-wing takeoff preserves the model ground offset rather than rejecting the model center as terrain slope',()=>{
 const {c,s,step}=fixture('rq180',{flight},13.4),m=c.fly_to({...command(900),altitudeM:100});c.mission_surface({id,missionId:m.id,surface:s});step();
 const v=c.observe(id);assert.equal(v.mission.status,'running');assert.equal(v.position.z,13.4);assert.ok(v.speedMps>0);
});

test('boat can turn back to a nearby departure outside its current turning circle',()=>{
 const {c,s,step}=fixture('patrol-boat',{boat:{maxSpeedMs:18,accelMs2:4.5,rudderTurnRadS2:.9,yawDamping:1.05}},.5);
 c.get(id).headingRad=335*Math.PI/180;const p={...command(27.18923),autoReturn:true};p.destination.lon+=12.67855/(6378137*Math.cos(64*Math.PI/180))*180/Math.PI;
 const m=c.sail_to(p);c.mission_surface({id,missionId:m.id,surface:s});let leg=m.id;
 for(let i=0;i<12000;i++){step();const v=c.observe(id);if(v.mission.id!==leg){leg=v.mission.id;c.mission_surface({id,missionId:leg,surface:s});}if(v.mission.status==='completed')break;}
 const v=c.observe(id);assert.equal(v.mission.leg,'return');assert.equal(v.mission.status,'completed');assert.ok(v.mission.remainingM<1.5);
});
