import test from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {groundHazard,groundSegmentClear} from '../src/ground-surface.mjs';
import {GroundControls} from '../src/ground-controls.mjs';
import {groundPerformance,groundProfile} from '../src/vehicle-performance.mjs';
const plane=grade=>({origin:{lat:0,lon:0},minX:0,minY:0,stepM:2,rows:5,cols:5,heights:Array.from({length:25},(_,i)=>10+(i%5)*2*grade),obstacles:[]});
test('manufacturer grades distinguish climbing from side slope; percentages are not degrees',()=>{
 const s=plane(.6),p={x:4,y:4};
 assert.equal(groundHazard(s,p,-Math.PI/2),null);
 assert.equal(groundHazard(s,p,0),'terrain-side-slope-exceeds-profile');
 assert.equal(groundHazard(plane(.8),p,-Math.PI/2),'terrain-climb-exceeds-profile');
});
test('route edges check the terrain between valid grid endpoints',()=>{
 const s=plane(0);s.heights=Array.from({length:25},(_,i)=>(i%5)>=2?12.6:10);
 assert.equal(groundHazard(s,{x:2,y:4},-Math.PI/2),null);
 assert.equal(groundHazard(s,{x:4,y:4},-Math.PI/2),null);
 assert.equal(groundSegmentClear(s,{x:2,y:4},{x:4,y:4}),false);
});
function setup(targetM){
 let now=0;const c=new GroundControls([],()=>now),id=randomUUID(),ownerAccountId=randomUUID();
 const s={origin:{lat:0,lon:0},minX:-2,minY:0,stepM:2,rows:1001,cols:3,heights:Array(3003).fill(10),obstacles:[],roads:[]};
 c.attach({id,ownerAccountId,terrainAssetId:'test-vehicle',definitionId:'patria-amv',pose:{x:0,y:0,headingRad:0},surface:s});
 const m=c.drive_to({id,ownerAccountId,actor:'test',requestId:randomUUID(),destination:{lat:targetM/6378137*180/Math.PI,lon:0}});
 c.mission_surface({id,missionId:m.id,surface:s});
 return {c,id,step(){now+=1000/30;c.step(1/30);}};
}
test('AMV accelerates beyond the old demo cap and stops at its coordinate destination',()=>{
 const {c,id,step}=setup(1800);let peak=0;
 for(let i=0;i<6000&&c.get(id).mission.status!=='completed';i++){step();peak=Math.max(peak,c.get(id).speedMps);}
 const v=c.get(id),spec=groundPerformance('patria-amv');
 assert.equal(spec.published.maximumSpeedKph.value,100);assert.equal(spec.published.maximumSpeedKph.relation,'>');
 assert.ok(peak>27);assert.ok(peak<=100/3.6);
 assert.equal(v.mission.status,'completed');assert.ok(Math.abs(v.position.y-1800)<=1.5);assert.ok(Math.abs(v.speedMps)<.05);
 assert.equal(c.capabilities({id}).limits.maxSpeedMps,groundProfile('patria-amv').maxForwardMps);
 const saved=c.exportState();saved[0].mission.maxSpeedMps=4;
 assert.equal(new GroundControls(saved).get(id).mission.maxSpeedMps,100/3.6);
});

test('rolling terrain patches extend a straight route without repeated stops or reverse gear',()=>{
 let now=0;const c=new GroundControls([],()=>now),id=randomUUID(),ownerAccountId=randomUUID();
 const origin={lat:64,lon:-51},targetY=1200;
 const surface=y=>({origin,minX:-256,minY:Math.round(y/2)*2-256,stepM:2,rows:257,cols:257,heights:Array(257*257).fill(10),obstacles:[],roads:[]});
 c.attach({id,ownerAccountId,terrainAssetId:'continuous-road-test',definitionId:'patria-amv',pose:{x:0,y:0,headingRad:0},surface:surface(0)});
 c.drive_to({id,ownerAccountId,actor:'test',requestId:randomUUID(),destination:{lat:origin.lat+targetY/6378137*180/Math.PI,lon:origin.lon}});
 const v=c.get(id),route={complete:true,stepM:2,points:Array.from({length:600},(_,i)=>({x:0,y:(i+1)*2}))};
 c.mission_surface({id,missionId:v.mission.id,surface:surface(0),destinationRoute:route});
 let lastY=0,peak=0,refreshes=0;
 for(let i=0;i<30*240&&v.mission.status!=='completed';i++){
  if(v.position.y-lastY>=32){
   const prefix=structuredClone(v.mission.navigation.points);
   c.mission_surface({id,missionId:v.mission.id,surface:surface(v.position.y)});
   assert.deepEqual(v.mission.navigation.points.slice(0,prefix.length),prefix,'refresh preserves the route already being followed');
   lastY=v.position.y;refreshes++;
  }
  now+=1000/30;c.step(1/30);peak=Math.max(peak,v.speedMps);
  assert.ok(v.speedMps>=0,'open road must not need reverse');
  if(v.position.y>100&&v.position.y<targetY-100)assert.ok(v.speedMps>1,'must not stop at local planning boundaries');
  if(v.mission.status==='blocked')assert.fail(v.mission.reason);
 }
 assert.equal(v.mission.status,'completed');assert.ok(refreshes>20);assert.ok(peak>27&&peak<=100/3.6);
});
