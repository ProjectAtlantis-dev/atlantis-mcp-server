const {createRedisRealtimeBridgeFromEnv}=require('./create-redis-bridge');
const {createEnvelope}=require('../../protocol/src');

// Redis is delivery/replay only. World + bank SQLite remain the test authority.
async function createWorldRelay(env) {
    if (!env.GREENLAND_WORLD_ROOM_ID || !/^[A-Za-z0-9_-]{1,128}$/.test(env.GREENLAND_WORLD_ROOM_ID)) {
        throw Error('Explicit GREENLAND_WORLD_ROOM_ID is required for realtime relay');
    }
    const connection=createRedisRealtimeBridgeFromEnv({env});
    try {await connection.connect();}
    catch(error){await connection.disconnect();throw error;}
    const room=env.GREENLAND_WORLD_ROOM_ID;
    return {
        async publish(snapshot){
            // Delivery sequence survives publisher reconnect/restart; gaps are valid.
            const sequence=Number(await connection.publisher.incr(`${connection.config.namespace}room:${room}:sequence`));
            return connection.bridge.publish(room,createEnvelope({kind:'snapshot',roomId:room,
                sequence,tick:sequence,payload:{authority:'greenland_world',clock:'serverTime',...snapshot}}));
        },
        close:()=>connection.disconnect()
    };
}
module.exports={createWorldRelay};
