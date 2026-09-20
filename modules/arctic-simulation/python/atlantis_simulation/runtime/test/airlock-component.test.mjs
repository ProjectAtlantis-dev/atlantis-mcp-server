import test from 'node:test';
import assert from 'node:assert/strict';
import {airlockState,commandAirlock,stepAirlock} from '../src/airlock-component.mjs';
import {InfrastructureState} from '../src/infrastructure-state.mjs';
test('commands are version checked, interlocked and completed only by elapsed steps',()=>{
 const s=airlockState();commandAirlock(s,{action:'airlock_open_outer',expectedRevision:0});
 assert.equal(s.outer,0);assert.throws(()=>commandAirlock(s,{action:'airlock_open_inner',expectedRevision:1}),/Close/);
 stepAirlock(s,1);assert.equal(s.outer,.5);
 assert.throws(()=>commandAirlock(s,{action:'airlock_close',expectedRevision:0}),/Stale/);
 commandAirlock(s,{action:'airlock_close',expectedRevision:1});
 assert.throws(()=>commandAirlock(s,{action:'airlock_open_inner',expectedRevision:2}),/Close/);
 stepAirlock(s,1);commandAirlock(s,{action:'airlock_open_inner',expectedRevision:2});stepAirlock(s,2);assert.equal(s.inner,1);
});
test('checkpoint roundtrip preserves in-motion state; public placement cannot inject state',()=>{
 const state=new InfrastructureState();const entity=state.place({modelId:'future-airlock-pedestrian',position:{x:0,y:0,z:0},componentState:{outer:1}});
 assert.equal(entity.componentState.outer,0);
 state.command({id:entity.id,action:'airlock_open_outer',expectedRevision:0});state.step(.5);
 const restored=new InfrastructureState(JSON.parse(JSON.stringify(state.snapshot())));
 assert.deepEqual(restored.snapshot(),state.snapshot());restored.step(1.5);assert.equal(restored.snapshot()[0].componentState.outer,1);
 assert.throws(()=>airlockState({schema:1,revision:0,outer:1,inner:1,target:{outer:0,inner:0}}),/conflicting/);
});
