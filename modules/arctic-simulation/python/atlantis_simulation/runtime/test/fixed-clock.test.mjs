import test from 'node:test';
import assert from 'node:assert/strict';
import {FixedClock} from '../src/fixed-clock.mjs';
test('long stalls retain elapsed time while each catch-up pump is bounded',()=>{
 const clock=new FixedClock();let elapsed=0;
 const first=clock.advance(2,dt=>elapsed+=dt);
 assert.equal(first.steps,8);assert.ok(first.pendingSeconds>1.7);
 while(clock.pendingSeconds>=clock.stepSeconds-1e-12)clock.advance(0,dt=>elapsed+=dt);
 assert.ok(Math.abs(elapsed-2)<1e-10);assert.ok(clock.pendingSeconds<1e-10);
});
test('callback failures retain unprocessed time',()=>{
 const clock=new FixedClock();assert.throws(()=>clock.advance(1,()=>{throw Error('failed');}));
 assert.equal(clock.pendingSeconds,1);
});
