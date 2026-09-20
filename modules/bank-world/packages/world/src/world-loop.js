// Timestamp-based logistics progression, independent of viewers and tool calls.
// One in-flight update at a time. Errors halt the loop and must be surfaced.
function createWorldLoop({world, intervalMs=100, onError, afterStep=null}) {
    if (!world || typeof world.advanceWorld !== 'function' || typeof onError !== 'function') {
        throw new TypeError('world and explicit onError handler are required');
    }
    let timer=null, pending=null, failure=null;
    function pump() {
        if (pending || failure) return;
        pending=Promise.resolve().then(async()=>{
            await world.advanceWorld();
            if(afterStep) await afterStep();
        }).catch(error=>{
            failure=error;
            clearInterval(timer); timer=null;
            onError(error);
        }).finally(()=>{pending=null;});
    }
    return {
        start(){if(timer || failure) throw Error('world loop already started or failed');timer=setInterval(pump,intervalMs);pump();},
        async stop(){clearInterval(timer);timer=null;if(pending)await pending;},
        get failure(){return failure;}
    };
}
module.exports={createWorldLoop};
