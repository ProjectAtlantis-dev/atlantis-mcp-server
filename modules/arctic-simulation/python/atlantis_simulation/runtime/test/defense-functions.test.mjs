import test from 'node:test';
import assert from 'node:assert/strict';
import {SimulationRoom} from '../src/engine.mjs';
function room(){return new SimulationRoom('fictional-demo',{automaticDefense:false,sites:[{id:'game-site',position:{x:0,y:0,z:0},sensorRangeM:1000,trackBuildSeconds:.1,layers:[{id:'toy-layer',targetKinds:['drone'],minRangeM:0,maxRangeM:1000,minAltitudeM:0,maxAltitudeM:1000,inventory:3,effectiveness:1,interceptorSpeedMps:1000,fuseRadiusM:20}]}]});}
function target(r){return r.spawnTarget({id:'test-target',kind:'drone',start:{x:100,y:0,z:10},destination:{x:0,y:0,z:10},speedMps:1});}
test('function mode tracks without launching; explicit observed choice launches once and resolves on ticks',()=>{
 const r=room();target(r);assert.equal(r.defenseObservation().tracks.length,0);
 assert.equal(r.commandIntercept({targetId:'test-target',siteId:'game-site',layerId:'toy-layer'},{queueIfUntracked:false}).reason,'target-not-tracked');
 r.step(.2);const observation=r.defenseObservation();assert.equal(observation.mode,'functions');assert.equal(observation.tracks[0].state,'tracked');assert.equal(r.engagements.size,0);
 assert.deepEqual(observation.tracks[0].availableActions,[{siteId:'game-site',layerId:'toy-layer',kind:'interceptor'}]);
 assert.equal(r.commandIntercept({targetId:'test-target',siteId:'wrong',layerId:'toy-layer'}).accepted,false);
 assert.equal(r.commandIntercept({targetId:'test-target',siteId:'game-site',layerId:'toy-layer'}).accepted,true);
 assert.equal(r.commandIntercept({targetId:'test-target',siteId:'game-site',layerId:'toy-layer'}).reason,'already-engaged');
 for(let i=0;i<30;i++)r.step(.1);assert.equal(r.targets.get('test-target').status,'intercepted');
 assert.ok(r.events.some(e=>e.type==='target-intercepted'));
});
test('mode changes preserve assets, clear queued orders, persist and reject invalid mode',()=>{
 const r=room();target(r);r.infrastructure.place({id:'pad',modelId:'network-radome',position:{x:1,y:2,z:0}});
 r.commandIntercept({targetId:'test-target'});assert.equal(r.interceptOrders.size,1);
 const change=r.setDefenseMode('functions');assert.equal(change.cancelledOrders,1);assert.equal(r.infrastructure.entities.size,1);
 const restored=new SimulationRoom(r.id,{},r.exportState());assert.equal(restored.defenseObservation().mode,'functions');assert.equal(restored.interceptOrders.size,0);
 assert.throws(()=>r.setDefenseMode('typo'),/mode/);r.setDefenseMode('automatic');r.step(.2);assert.equal(r.engagements.size,1);
});
test('every configured game layer has a typed fixture that radar detects without activating it',()=>{
 const r=new SimulationRoom('layer-fixtures',{automaticDefense:false});
 for(const site of r.config.sites)for(const layer of site.layers){
  const args={incomingType:layer.targetKinds[0],requestId:site.id+'-'+layer.id,siteId:site.id,testLayerId:layer.id};
  const result=r.spawnTestIncoming(args);assert.equal(result.alreadyExists,false);
  assert.equal(r.spawnTestIncoming(args).alreadyExists,true);
  r.step(.5);const track=r.defenseObservation().tracks.find(t=>t.id===result.target.id);
  assert.equal(track.state,'tracked');assert.ok(track.detectedBy.length>0);
  assert.ok(track.availableActions.some(a=>a.layerId===layer.id&&a.siteId===site.id));
  assert.equal(r.engagements.size,0);
 }
 assert.throws(()=>r.spawnTestIncoming({incomingType:'drone',requestId:'bad',siteId:r.config.sites[0].id,testLayerId:'upper-tier'}),/not supported/);
 const first=r.targets.values().next().value;
 assert.throws(()=>r.spawnTestIncoming({incomingType:'drone',requestId:first.id.slice('test-incoming:'.length),siteId:r.config.sites[0].id,testLayerId:'point-defense'}),/different test terms/);
});
test('offline radar cannot detect; loss of an existing track is explicit in events',()=>{
 const r=room();target(r);const radar=[...r.vehicleSystem.vehicles.values()].find(v=>v.role==='sensor');
 radar.status='lost';r.step(.2);assert.deepEqual(r.defenseObservation().tracks,[]);assert.equal(r.defenseObservation().sensors[0].operational,false);
 radar.status='deployed';r.step(.2);assert.equal(r.defenseObservation().tracks[0].state,'tracked');
 assert.ok(r.events.some(e=>e.type==='target-detected'));assert.ok(r.events.some(e=>e.type==='target-tracked'));
 radar.status='lost';r.step(.2);assert.deepEqual(r.defenseObservation().tracks,[]);assert.ok(r.events.some(e=>e.type==='target-track-lost'));
});

test('incoming heading controls actual travel and is preserved in observed type/status; invalid input rejects',()=>{
 const r=room();
 for(const [heading,dx,dy] of [[0,0,1],[90,1,0],[180,0,-1],[270,-1,0]]){
  const result=r.spawnTestIncoming({incomingType:'drone',requestId:'heading-'+heading,siteId:'game-site',testLayerId:'toy-layer',headingDeg:heading});
  const v=result.target.velocity;assert.ok(Math.abs(v.x/result.target.speedMps-dx)<1e-8);assert.ok(Math.abs(v.y/result.target.speedMps-dy)<1e-8);
  r.step(.2);const track=r.defenseObservation().tracks.find(t=>t.id===result.target.id);assert.equal(track.headingDeg,heading);assert.equal(track.kind,'drone');
 }
 assert.throws(()=>r.spawnTestIncoming({incomingType:'drone',requestId:'bad-heading',siteId:'game-site',testLayerId:'toy-layer',headingDeg:360}),/headingDeg/);
});
test('coordinate incoming travels toward the supplied destination, preserves retry identity and never selects a layer',()=>{
 const r=room(),args={incomingType:'drone',requestId:'coordinate-test',destination:{x:20,y:30,z:5},destinationCoordinates:{latitude:64,longitude:-51},headingDeg:270,approachDistanceM:100,altitudeM:50,speedMps:10};
 const result=r.spawnIncoming(args);assert.ok(result.target.start.x>args.destination.x);assert.ok(result.target.velocity.x<0);assert.deepEqual(result.target.destination,args.destination);
 assert.equal(r.spawnIncoming(args).alreadyExists,true);
 assert.throws(()=>r.spawnIncoming({...args,headingDeg:90}),/different scenario terms/);
 for(let i=0;i<120;i++)r.step(.1);
 assert.equal(r.targets.get(result.target.id).status,'leaked');assert.equal(r.engagements.size,0);
 assert.deepEqual(r.targets.get(result.target.id).position,args.destination);
});
