const test=require('node:test');
const assert=require('node:assert/strict');
const {createWorldLoop}=require('../packages/world/src/world-loop');

test('world advances without viewers, avoids overlapping writes, drains on stop',async()=>{
    let active=0,max=0,count=0;
    const loop=createWorldLoop({intervalMs:2,onError:error=>{throw error;},world:{async advanceWorld(){
        active++;max=Math.max(max,active);count++;
        await new Promise(resolve=>setTimeout(resolve,12));active--;
    }}});
    loop.start();await new Promise(resolve=>setTimeout(resolve,60));await loop.stop();
    assert.ok(count>=1);assert.equal(max,1);assert.equal(active,0);
    const stopped=count;await new Promise(resolve=>setTimeout(resolve,15));assert.equal(count,stopped);
});

test('world loop surfaces failure and stops instead of silently continuing',async()=>{
    let reported;
    const error=Error('storage unavailable');
    const loop=createWorldLoop({intervalMs:2,world:{async advanceWorld(){throw error;}},onError:e=>{reported=e;}});
    loop.start();await new Promise(resolve=>setTimeout(resolve,20));await loop.stop();
    assert.equal(reported,error);assert.equal(loop.failure,error);assert.throws(()=>loop.start(),/failed/);
});
