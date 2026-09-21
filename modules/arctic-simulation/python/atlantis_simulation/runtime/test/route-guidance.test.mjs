import test from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {GroundControls} from '../src/ground-controls.mjs';
import {updateDestinationProgress} from '../src/route-guidance.mjs';
function fixture(){
 const c=new GroundControls(),id=randomUUID(),ownerAccountId=randomUUID();
 const surface={origin:{lat:64,lon:-51},minX:-128,minY:-128,stepM:2,rows:129,cols:129,heights:Array(129*129).fill(10),water:Array(129*129).fill(false),roads:[],obstacles:[]};
 c.attach({id,ownerAccountId,terrainAssetId:'test-ground',definitionId:'patria-amv',pose:{x:0,y:0,headingRad:-Math.PI/2},surface});
 c.drive_to({id,ownerAccountId,actor:'test',requestId:randomUUID(),destination:{lat:64,lon:-51+300/(6378137*Math.cos(64*Math.PI/180))*180/Math.PI}});
 const v=c.get(id),route={complete:true,stepM:2,points:Array.from({length:150},(_,i)=>({x:(i+1)*2,y:0}))};
 const supply=extra=>c.mission_surface({id,missionId:v.mission.id,surface,...extra});
 return {c,v,surface,route,supply};
}
test('remote dispatch requires a complete route and keeps its destination across local refreshes and restart',()=>{
 const {c,v,route,supply}=fixture();
 supply({destinationRoute:route});
 assert.equal(v.mission.navigation.complete,false);
 assert.ok(v.mission.navigation.points.at(-1).x<128);
 v.position.x=35;supply({});
 assert.ok(v.mission.destinationRoute.index>0);
 assert.equal(v.mission.destinationRoute.points.at(-1).x,300);
 const restored=new GroundControls(c.exportState()).get(v.id);
 assert.deepEqual(restored.mission.destinationRoute,v.mission.destinationRoute);
 assert.equal(restored.missionTerrainReady,false);
});
test('new local blockage requests destination replanning; failed replacement reports an explicit block',()=>{
 const {v,surface,route,supply}=fixture();supply({destinationRoute:route});
 surface.obstacles=[{minX:10,maxX:20,minY:-128,maxY:128}];
 supply({});assert.equal(v.mission.status,'queued');assert.equal(v.mission.routeNeedsReplan,true);assert.equal(v.missionTerrainReady,false);
 assert.match(v.mission.reason,/replanning-route/);
 supply({destinationRoute:route});assert.equal(v.mission.status,'blocked');assert.match(v.mission.reason,/no traversable route/);
});
test('route progress uses the driven corridor rather than demanding exact grid vertices',()=>{
 const {v,route,supply}=fixture();supply({destinationRoute:route});v.position={x:50,y:5,z:10};updateDestinationProgress(v);
 assert.ok(v.mission.destinationRoute.index>=24);
});
