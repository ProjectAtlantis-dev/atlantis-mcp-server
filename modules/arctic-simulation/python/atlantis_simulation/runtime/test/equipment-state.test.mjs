import test from 'node:test';import assert from 'node:assert/strict';
import {ASSET_CONTROLS,equipmentState,commandEquipment,stepEquipment} from '../src/equipment-state.mjs';
import {SimulationRoom} from '../src/engine.mjs';
test('every asset mechanism validates, advances on server ticks and survives persistence',()=>{
 assert.equal(Object.keys(ASSET_CONTROLS.models).length,89);
 for(const [model,c] of Object.entries(ASSET_CONTROLS.models)){
  const state=equipmentState(model);
  for(const m of c.mechanisms){
   const values=Object.fromEntries(Object.entries(m.fields).map(([key,s])=>[key,s.default]));
   commandEquipment(model,state,{mechanismId:m.id,values,expectedRevision:0});
   assert.throws(()=>commandEquipment(model,state,{mechanismId:m.id,values,expectedRevision:0}),/revision/);
   assert.throws(()=>commandEquipment(model,state,{mechanismId:m.id,values:{invented:1},expectedRevision:1}),/parameters/);
  }
  stepEquipment(model,state,1);assert.deepEqual(equipmentState(model,state),state);
 }
});
test('nested airlocks and freight retain interlocks throughout motion',()=>{
 const model='future-city-demo',c=ASSET_CONTROLS.models[model],state=equipmentState(model);
 const door=c.mechanisms.find(m=>m.kind==='airlock');
 commandEquipment(model,state,{mechanismId:door.id,values:{outer_open:true},expectedRevision:0});stepEquipment(model,state,.2);
 assert.throws(()=>commandEquipment(model,state,{mechanismId:door.id,values:{outer_open:false,inner_open:true},expectedRevision:1}),/Close/);
 commandEquipment(model,state,{mechanismId:door.id,values:{outer_open:false},expectedRevision:1});stepEquipment(model,state,2);
 commandEquipment(model,state,{mechanismId:door.id,values:{inner_open:true},expectedRevision:2});stepEquipment(model,state,2);
 assert.equal(state.mechanisms[door.id].actual.inner_open,1);
});
test('instance state is isolated; restoration preserves actual pose and target',()=>{
 const room=new SimulationRoom('rigs'),modelId='support-snowcat-plow',owner='11111111-1111-4111-8111-111111111111';
 for(const id of ['one','two'])room.infrastructure.place({id,modelId,position:{x:0,y:0,z:0},accessPolicy:{version:1,ownerAccountId:owner,allowedAccountIds:[],interactionRadiusM:5,bounds:{minX:-1,maxX:1,minY:-1,maxY:1,minZ:0,maxZ:3}}});
 room.infrastructure.equipment({id:'one',mechanismId:'plow',values:{lift_deg:18},expectedRevision:0},{accountId:owner});
 room.infrastructure.step(.5);
 assert.equal(room.infrastructure.entities.get('one').equipmentState.mechanisms.plow.actual.lift_deg,4.5);
 assert.equal(room.infrastructure.entities.get('two').equipmentState.mechanisms.plow.actual.lift_deg,0);
 const restored=new SimulationRoom('rigs',{},room.exportState());assert.deepEqual(restored.snapshot().infrastructure,room.snapshot().infrastructure);
 assert.throws(()=>room.infrastructure.equipment({id:'one',mechanismId:'lights',values:{enabled:false},expectedRevision:0},{accountId:'other'}),/owner/);
});
