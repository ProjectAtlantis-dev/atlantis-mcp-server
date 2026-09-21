import test from 'node:test';
import assert from 'node:assert/strict';
import {stepBoat} from '../src/boat-controls.mjs';

test('dense route bends do not force a boat to circle around individual vertices',()=>{
 const v={position:{x:0,y:0,z:.5},headingRad:0,speedMps:0,yawRate:0,distanceM:0,signedDistanceM:0,revision:0,waterLevelM:.5,missionTerrainReady:true,
  boatProfile:{maxSpeedMs:18,accelMs2:4.5,rudderTurnRadS2:.9,yawDamping:1.05},
  surface:{origin:{lat:0,lon:0},navigationDomain:'water',minX:-200,minY:-200,rows:201,cols:201,stepM:2,heights:Array(40401).fill(.5),water:Array(40401).fill(true)},
  mission:{status:'running',target:{x:100,y:100},arrivalToleranceM:1.5,startDistanceM:0,lastProgressAt:0,navigation:{index:0,complete:true,points:[{x:0,y:20},{x:4,y:24},{x:8,y:28},{x:12,y:32},{x:16,y:36},{x:20,y:40},{x:60,y:60},{x:100,y:100}]}}};
 let turn=0,previous=0;
 for(let i=0;i<30*30&&v.mission.status==='running';i++){
  stepBoat(v,1/30,i*1000/30);turn+=Math.abs(v.headingRad-previous);previous=v.headingRad;
 }
 assert.equal(v.mission.status,'completed');
 assert.ok(v.distanceM<200,`Unnecessary route travel: ${v.distanceM}`);
 assert.ok(turn<Math.PI,`Unnecessary circling: ${turn}`);
});

import {planBoatManeuver} from '../src/boat-maneuver.mjs';
test('a boat facing a shoreline can plan a verified reverse departure',()=>{
 const water=Array.from({length:101*101},(_,i)=>-100+(i%101)*2<=0);
 const v={position:{x:0,y:0,z:.5},headingRad:-Math.PI/2,
  boatProfile:{maxSpeedMs:18,reverseMaxSpeedMs:5,rudderTurnRadS2:.9,yawDamping:1.05},
  surface:{origin:{lat:0,lon:0},navigationDomain:'water',minX:-100,minY:-100,rows:101,cols:101,stepM:2,heights:Array(10201).fill(.5),water}};
 const plan=planBoatManeuver(v,{x:-30,y:0});
 assert.ok(plan?.length,'No water escape trajectory found');
 assert.ok(plan.some(s=>s.gear===-1),'Recovery must use the existing reverse capability');
 assert.ok(plan.reduce((sum,s)=>sum+s.length,0)<60,'Recovery should not require a full circle');
});

test('an intermediate point inside the turning circle triggers a short maneuver and route extension',()=>{
const v={position:{x:0,y:0,z:.5},headingRad:0,speedMps:0,yawRate:0,distanceM:0,signedDistanceM:0,revision:0,waterLevelM:.5,missionTerrainReady:true,
boatProfile:{maxSpeedMs:18,reverseMaxSpeedMs:5,accelMs2:4.5,rudderTurnRadS2:.9,yawDamping:1.05},
surface:{origin:{lat:0,lon:0},navigationDomain:'water',minX:-100,minY:-100,rows:101,cols:101,stepM:2,heights:Array(10201).fill(.5),water:Array(10201).fill(true)},
mission:{status:'running',target:{x:500,y:0},arrivalToleranceM:1.5,startDistanceM:0,lastProgressAt:0,navigation:{index:0,complete:false,points:[{x:10,y:0},{x:20,y:0}]}}};
for(let i=0;i<120*30;i++){stepBoat(v,1/30,i*1000/30);if(!v.missionTerrainReady||v.mission.status==='blocked')break;}
assert.equal(v.mission.status,'running');
assert.equal(v.missionTerrainReady,false);
assert.equal(v.mission.reason,'awaiting-water-route-extension');
assert.ok(v.distanceM<100);
});

function routeFixture(){
const v={position:{x:0,y:0,z:.5},headingRad:0,speedMps:0,yawRate:0,distanceM:0,signedDistanceM:0,revision:0,waterLevelM:.5,missionTerrainReady:true,
boatProfile:{maxSpeedMs:18,reverseMaxSpeedMs:5,accelMs2:4.5,rudderTurnRadS2:.9,yawDamping:1.05},
surface:{origin:{lat:0,lon:0},navigationDomain:'water',minX:-100,minY:-100,rows:101,cols:101,stepM:2,heights:Array(10201).fill(.5),water:Array(10201).fill(true)},
mission:{status:'running',target:{x:500,y:0},arrivalToleranceM:1.5,startDistanceM:0,lastProgressAt:0,navigation:{index:0,complete:false,points:[{x:10,y:0},{x:20,y:0}]}}};
return v;
}

test('a passed first waypoint is skipped after a route refresh',()=>{
 const v=routeFixture();v.position.x=15;v.headingRad=-Math.PI/2;
 stepBoat(v,1/30,0);
 assert.equal(v.mission.navigation.index,1);
 assert.equal(v.mission.status,'running');
 assert.ok(v.speedMps>0);
});
test('an already reached maneuver target is success rather than a blocked mission',()=>{
 const v=routeFixture();v.headingRad=-Math.PI/2;v.mission.boatTurnTarget={x:0,y:0};
 stepBoat(v,1/30,0);
 assert.equal(v.mission.status,'running');
 assert.equal(v.mission.boatManeuver,undefined);
 assert.equal(v.mission.boatTurnTarget,undefined);
});
