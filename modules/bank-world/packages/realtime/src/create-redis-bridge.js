const Redis = require('ioredis');
const { validateRedisTestConfig } = require('../../persistence/src/redis-test-config');
const { RedisRealtimeBridge } = require('./redis-bridge');

function createRedisRealtimeBridgeFromEnv({ env = process.env, RedisImpl = Redis } = {}) {
    if (!env.GREENLAND_REDIS_TEST_URL || !env.GREENLAND_REDIS_NAMESPACE) {
        throw new Error('Explicit dedicated GREENLAND_REDIS_TEST_URL and GREENLAND_REDIS_NAMESPACE are required');
    }
    const config = validateRedisTestConfig({
        url: env.GREENLAND_REDIS_TEST_URL,
        namespace: env.GREENLAND_REDIS_NAMESPACE,
        allowRemote: env.GREENLAND_ALLOW_REMOTE_TEST_REDIS === '1'
    });
    const connectionOptions = {
        lazyConnect: true,
        enableOfflineQueue: false,
        maxRetriesPerRequest: 1,
        retryStrategy(times) { return Math.min(times * 100, 2000); }
    };
    const publisher = new RedisImpl(config.url, {
        ...connectionOptions,
        connectionName: 'greenland-test-publisher'
    });
    const subscriber = new RedisImpl(config.url, {
        ...connectionOptions,
        connectionName: 'greenland-test-subscriber'
    });
    const bridge = new RedisRealtimeBridge({
        publisher,
        subscriber,
        namespace: config.namespace
    });

    async function connect() {
        await Promise.all([publisher.connect(), subscriber.connect()]);
        await bridge.start();
        return bridge;
    }

    async function disconnect() {
        await bridge.stop();
        await Promise.all([
            publisher.status === 'ready' ? publisher.quit() : publisher.disconnect(),
            subscriber.status === 'ready' ? subscriber.quit() : subscriber.disconnect()
        ]);
    }

    return { bridge, publisher, subscriber, config, connect, disconnect };
}

module.exports = { createRedisRealtimeBridgeFromEnv };
