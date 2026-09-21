import test from 'node:test';import assert from 'node:assert/strict';import {randomUUID} from 'node:crypto';
import {GroundControls} from '../src/ground-controls.mjs';
test('AMV completes a turn on a hillside without alternating gear-change braking forever',()=>{
 let now=Date.now();const c=new GroundControls([],()=>now),id=randomUUID(),ownerAccountId=randomUUID();
 const s={origin:{lat:64,lon:-51},minX:-60,minY:-60,stepM:2,rows:61,cols:61,heights:Array.from({length:61*61},(_,i)=>100+(-60+(i%61)*2)*.45),water:Array(61*61).fill(false),obstacles:[],roads:[]};
 c.attach({id,ownerAccountId,terrainAssetId:'maneuver-test',definitionId:'patria-amv',pose:{x:0,y:0,headingRad:10.3},surface:s});
 c.drive_to({id,actor:'test',ownerAccountId,requestId:randomUUID(),destination:{lat:64+16/6378137*180/Math.PI,lon:-51+14/(6378137*Math.cos(64*Math.PI/180))*180/Math.PI}});
 const v=c.get(id);c.mission_surface({id,missionId:v.mission.id,surface:s});
 for(let i=0;i<30*180&&v.mission.status!=='completed';i++){now+=1000/30;c.step(1/30);if(v.mission.status==='blocked')break;}
 assert.equal(v.mission.status,'completed',v.mission.reason);assert.ok(v.mission.remainingM<=1.5);
});
