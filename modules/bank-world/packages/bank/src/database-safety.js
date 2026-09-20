const LOCAL_TEST_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);

function validateTestDatabaseUrl(value, { allowRemote = false } = {}) {
    if (!value) throw new Error('GAME_BANK_TEST_DATABASE_URL is required');
    let url;
    try {
        url = new URL(value);
    } catch (_error) {
        throw new Error('GAME_BANK_TEST_DATABASE_URL must be a valid PostgreSQL URL');
    }
    if (!['postgres:', 'postgresql:'].includes(url.protocol)) {
        throw new Error('The test database URL must use postgres:// or postgresql://');
    }
    const databaseName = url.pathname.replace(/^\//, '');
    if (!databaseName.endsWith('_test')) {
        throw new Error('The test database name must end with _test');
    }
    if (!allowRemote && !LOCAL_TEST_HOSTS.has(url.hostname)) {
        throw new Error('Remote test databases require an explicit allowRemote opt-in');
    }
    return {
        hostname: url.hostname,
        port: url.port || '5432',
        databaseName,
        local: LOCAL_TEST_HOSTS.has(url.hostname)
    };
}

module.exports = { validateTestDatabaseUrl };
