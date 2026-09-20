const path = require('node:path');

const LOCAL_TEST_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);
const FORBIDDEN_DATABASE_URLS = ['DATABASE_URL', 'BANK_DATABASE_URL', 'WORLD_DATABASE_URL'];

function validateSqliteTestPath(value, name) {
    if (typeof value !== 'string' || !value.trim()) throw new Error(`${name} is required`);
    const normalized = path.resolve(value);
    if (!normalized.endsWith('.test.sqlite')) {
        throw new Error(`${name} must end with .test.sqlite`);
    }
    return normalized;
}

function validatePostgresTestUrl(value, { allowRemote = false } = {}) {
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
    return { hostname: url.hostname, port: url.port || '5432', databaseName };
}

function validateOptionalPostgresTestUrl(env = process.env) {
    if (!env.GAME_BANK_TEST_DATABASE_URL) return null;
    return validatePostgresTestUrl(env.GAME_BANK_TEST_DATABASE_URL, {
        allowRemote: env.GAME_BANK_ALLOW_REMOTE_TEST_DB === '1'
    });
}

function assertTestOnlyRuntime({ env = process.env, bankDbPath, worldDbPath }) {
    if (env.NODE_ENV === 'production') {
        throw new Error('This foundation is test-only and refuses NODE_ENV=production');
    }
    for (const name of FORBIDDEN_DATABASE_URLS) {
        if (env[name]) throw new Error(`${name} is forbidden in the test-only SQLite runtime`);
    }
    return {
        bankDbPath: validateSqliteTestPath(bankDbPath, 'BANK_DB_PATH'),
        worldDbPath: validateSqliteTestPath(worldDbPath, 'WORLD_DB_PATH')
    };
}

module.exports = {
    assertTestOnlyRuntime,
    validateOptionalPostgresTestUrl,
    validatePostgresTestUrl,
    validateSqliteTestPath
};
