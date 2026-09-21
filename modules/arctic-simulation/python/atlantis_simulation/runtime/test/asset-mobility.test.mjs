import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {GroundControls} from '../src/ground-controls.mjs';
const models=JSON.parse(fs.readFileSync(new URL('../src/asset-mobility.json',import.meta.url)));
for(const [definitionId,profile] of Object.entries(models))test(definitionId+' reaches a coordinate and survives restoration',()=>{
 const id='12345678-1234-4234-8234-123456789abc',ownerAccountId='22345678-1234-4234-8234-123456789abc';
 const water=profile.domain==='water';
 const surface={origin:{lat:64,lon:-51},minX:-200,minY:-200,stepM:2,rows:201,cols:201,heights:Array(40401).fill(water?0:10),water:Array(40401).fill(water)};
 let now=0;const c=new GroundControls([],()=>now);
 c.attach({id,ownerAccountId,definitionId,terrainAssetId:id,pose:{x:0,y:0,headingRad:0},sourcePose:{z:0},surface});
 const m=c[water?'sail_to':'drive_to']({id,ownerAccountId,actor:'test',requestId:'one',destination:{lat:64+50/6378137*180/Math.PI,lon:-51}});
 c.mission_surface({id,missionId:m.id,surface});
 for(let i=0;i<6000&&c.observe(id).mission.status!=='completed';i++){now+=1000/30;c.step(1/30);}
 const state=c.observe(id);assert.equal(state.mission.status,'completed');assert.ok(state.mission.remainingM<=1.5);assert.ok(state.distanceM>48);
 const restored=new GroundControls(c.exportState()).observe(id);assert.equal(restored.definitionId,definitionId);assert.deepEqual(restored.position,state.position);
});
