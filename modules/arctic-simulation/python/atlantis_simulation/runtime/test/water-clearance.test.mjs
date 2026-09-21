import test from 'node:test';
import assert from 'node:assert/strict';
import {planTerrainRoute} from '../src/terrain-route.mjs';
import {groundHazard} from '../src/ground-surface.mjs';
import {waterClearance,waterShortcutClear} from '../src/water-clearance.mjs';
function surface(){const cols=101,rows=81;return {cols,rows,minX:0,minY:0,stepM:2,heights:Array(cols*rows).fill(0),water:Array.from({length:cols*rows},(_,i)=>Math.floor(i/cols)>10),navigationDomain:'water',waterNavigation:{hullRadiusM:6,preferredClearanceM:42}};}
test('water routing leaves shore for open water instead of shortest coastal line',()=>{
 const s=surface(),a={x:20,y:32},b={x:180,y:32};const route=planTerrainRoute(s,a,b,{dense:true});
 assert.ok(route.complete);assert.ok(Math.max(...route.points.map(p=>p.y))>=58);
 assert.ok(route.points.every(p=>!groundHazard(s,p)));assert.equal(route.mode,'water');
});
test('hull clearance rejects a centre on water with hull overlapping shore',()=>{
 const s=surface();assert.equal(s.water[12*s.cols+20],true);assert.equal(groundHazard(s,{x:40,y:24}),'insufficient-hull-clearance');
 assert.throws(()=>planTerrainRoute(s,{x:20,y:40},{x:180,y:24}),/destination is not traversable/);
});
test('water shortcut cannot cut inside clearance around a headland',()=>{
 const s=surface();for(let y=0;y<50;y++)for(let x=48;x<=52;x++)s.water[y*s.cols+x]=false;
 const a={x:30,y:110},b={x:170,y:110};assert.ok(waterClearance(s,a)>42);assert.equal(waterShortcutClear(s,a,b),false);
});
test('hull-safe narrow waterway remains usable, narrower one is rejected',()=>{
 const s=surface();s.water=s.water.map((_,i)=>{const y=Math.floor(i/s.cols);return y>28&&y<42;});
 assert.ok(planTerrainRoute(s,{x:20,y:70},{x:180,y:70}).complete);
 s.water=s.water.map((_,i)=>{const y=Math.floor(i/s.cols);return y>33&&y<37;});
 // Surfaces are immutable: make a new object after replacing its water mask.
 assert.throws(()=>planTerrainRoute({...s},{x:20,y:70},{x:180,y:70}),/outside traversable/);
});
test('shoreline preference permits grid-resolution variation along a safe route edge',()=>{
 const s=surface();assert.equal(waterShortcutClear(s,{x:20,y:60},{x:24,y:60}),true);
});
