import test from 'node:test';
import assert from 'node:assert/strict';
import {groundHazard,groundSegmentClear,surfaceHeight,surfaceNormal} from '../src/ground-surface.mjs';
import {planTerrainRoute} from '../src/terrain-route.mjs';
function crossing(){
 const rows=41,cols=41,heights=[],water=[];
 for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
  const wet=Math.abs(-40+row*2)<8;heights.push(wet?0:4);water.push(wet);
 }
 return {origin:{lat:64,lon:-51},minX:-40,minY:-40,stepM:2,rows,cols,heights,water,obstacles:[],roads:[{surfaceHalfWidthM:4,path:[{x:0,y:-30,z:4},{x:0,y:30,z:4}]}]};
}
test('surveyed crossing connects land areas and controller uses deck elevation',()=>{
 const s=crossing(),start={x:0,y:-30},end={x:0,y:30};
 const r=planTerrainRoute(s,start,end,{dense:true});assert.equal(r.complete,true);
 let a=start;for(const b of r.points){assert.equal(groundSegmentClear(s,a,b),true);a=b;}
 assert.equal(surfaceHeight(s,0,0),4);assert.equal(groundHazard(s,{x:0,y:0}),null);
 assert.equal(surfaceNormal(s,{x:0,y:0}).z,1);
 assert.match(groundHazard(s,{x:8,y:0}),/water/);
});
test('2D road preference alone cannot turn water into drivable land',()=>{
 const s=crossing();delete s.roads[0].surfaceHalfWidthM;
 assert.throws(()=>planTerrainRoute(s,{x:0,y:-30},{x:0,y:30}),/no traversable/);
});
test('surveyed roads do not bypass buildings, grades or missing elevations',()=>{
 const s=crossing();s.obstacles=[{minX:-6,maxX:6,minY:-3,maxY:3}];
 assert.equal(groundHazard(s,{x:0,y:0}),'building-clearance');
 const steep=crossing();steep.roads[0].path[1].z=120;
 assert.match(groundHazard(steep,{x:0,y:0}),/slope/);
 const missing=crossing();delete missing.roads[0].path[0].z;
 assert.throws(()=>surfaceHeight(missing,0,0),/surveyed road elevation/);
});
test('boat water surface is unaffected by a road deck',()=>{
 const s=crossing();s.navigationDomain='water';
 assert.equal(surfaceHeight(s,0,0),0);
});
