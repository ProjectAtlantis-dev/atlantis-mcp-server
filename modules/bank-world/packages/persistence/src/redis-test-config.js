const LOCAL_REDIS_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);

function validateRedisTestConfig({
    url = 'redis://127.0.0.1:6379/15',
    namespace = 'greenland:test:',
    allowRemote = false
} = {}) {
    let parsed;
    try {
        parsed = new URL(url);
    } catch (_error) {
        throw new Error('GREENLAND_REDIS_TEST_URL must be a valid Redis URL');
    }
    if (!['redis:', 'rediss:'].includes(parsed.protocol)) {
        throw new Error('GREENLAND_REDIS_TEST_URL must use redis:// or rediss://');
    }
    if (!allowRemote && !LOCAL_REDIS_HOSTS.has(parsed.hostname)) {
        throw new Error('Remote Redis requires GREENLAND_ALLOW_REMOTE_TEST_REDIS=1');
    }
    const database = Number((parsed.pathname || '/0').slice(1) || 0);
    if (!Number.isInteger(database) || database < 0) {
        throw new Error('Redis database index must be a non-negative integer');
    }
    if (typeof namespace !== 'string' || !namespace.startsWith('greenland:test:')) {
        throw new Error('Redis namespace must start with greenland:test:');
    }
    return {
        url: parsed.toString(),
        hostname: parsed.hostname,
        database,
        namespace,
        local: LOCAL_REDIS_HOSTS.has(parsed.hostname)
    };
}

module.exports = { validateRedisTestConfig };
