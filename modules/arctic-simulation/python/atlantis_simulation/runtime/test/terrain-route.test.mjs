import test from 'node:test';import assert from 'node:assert/strict';import {planTerrainRoute} from '../src/terrain-route.mjs';
const grid=()=>({origin:{lat:64,lon:-51},minX:-40,minY:-40,stepM:2,rows:41,cols:41,heights:Array(41*41).fill(10),obstacles:[],roads:[]});
test('route detours around building instead of cutting through it',()=>{const s=grid();s.obstacles=[{minX:-5,maxX:5,minY:5,maxY:15}];const r=planTerrainRoute(s,{x:0,y:0},{x:0,y:30});assert.ok(r.complete);assert.ok(r.points.some(p=>Math.abs(p.x)>5));});
test('road preference uses surveyed lines and offroad works without them',()=>{const s=grid();s.roads=[{path:[{x:0,y:0},{x:8,y:0},{x:8,y:30},{x:0,y:30}]}];const r=planTerrainRoute(s,{x:0,y:0},{x:0,y:30});assert.ok(r.roadCells>r.cells/2);assert.equal(planTerrainRoute(grid(),{x:0,y:0},{x:0,y:30}).complete,true);});
test('water barrier fails explicitly and remote target yields bounded partial path',()=>{const s=grid();s.water=Array(1681).fill(false);for(let y=23;y<27;y++)for(let x=0;x<41;x++)s.water[y*41+x]=true;assert.throws(()=>planTerrainRoute(s,{x:0,y:0},{x:0,y:30}),/no traversable/);const r=planTerrainRoute(grid(),{x:0,y:0},{x:0,y:200});assert.equal(r.complete,false);assert.ok(r.points.at(-1).y<40);});

test('planner accepts the same already padded building clearance as the controller',()=>{
 const s=grid();s.obstacles=[{minX:-30,maxX:-1.9,minY:-30,maxY:20}];
 const route=planTerrainRoute(s,{x:0,y:0},{x:0,y:30});assert.ok(route.complete);
 assert.throws(()=>planTerrainRoute(s,{x:-2,y:0},{x:0,y:30}),/starts outside/);
});
