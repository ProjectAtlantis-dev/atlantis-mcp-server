const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { assetIdentityPolicy, AUTHORITY } = require('../packages/domain/src');
const {
    bytesToUuid,
    createEnvelope,
    uuidToBytes,
    validateEnvelope
} = require('../packages/protocol/src');
const { AuthoritativeRoom } = require('../packages/realtime/src');
const { RedisRealtimeBridge } = require('../packages/realtime/src/redis-bridge');
const { LocalGeodeticFrame } = require('../packages/physics/src/local-frame');
const {
    assertTestOnlyRuntime,
    validatePostgresTestUrl
} = require('../packages/persistence/src/test-config');
const { validateRedisTestConfig } = require('../packages/persistence/src/redis-test-config');
const { createApplication } = require('../apps/api/src/server');

test('asset policy distinguishes serialized UUID instances from conserved lots', () => {
    assert.equal(assetIdentityPolicy({ kind: 'vehicle' }), 'one_uuid_per_instance');
    assert.equal(
        assetIdentityPolicy({ kind: 'resource_lot', quantity: 12.5, unit: 'tonnes' }),
        'one_uuid_per_conserved_lot'
    );
    assert.equal(AUTHORITY.money, 'greenland_bank');
    assert.throws(() => assetIdentityPolicy({ kind: 'vehicle', quantity: 2 }), /one instance/);
});

test('protocol envelopes carry ordered room/tick state and compact UUIDs round-trip', () => {
    const uuid = 'f9941849-c59f-49cf-9ee4-fc92e9d87cd7';
    assert.equal(bytesToUuid(uuidToBytes(uuid)), uuid);
    assert.deepEqual(validateEnvelope(createEnvelope({
        kind: 'snapshot',
        roomId: 'port:NUUK',
        sequence: 4,
        tick: 20,
        payload: { boats: [] }
    })).payload, { boats: [] });
    assert.throws(() => validateEnvelope({ version: 99 }), /Unsupported protocol version/);
});

test('authoritative rooms sequence messages and replay only missed state', () => {
    const room = new AuthoritativeRoom({ id: 'port:NUUK', maxReplayMessages: 2 });
    room.advanceTick();
    room.publish('event', { type: 'boat_entered' });
    room.publish('delta', { boat: 'one' });
    room.publish('delta', { boat: 'two' });
    assert.deepEqual(room.messagesAfter(1).map((message) => message.sequence), [2, 3]);
    assert.equal(room.messagesAfter(2)[0].tick, 1);
});

test('Nuuk local physics frame round-trips nearby WGS84 without giant coordinates', () => {
    const frame = new LocalGeodeticFrame({ lat: 64.1797, lon: -51.7414, altitude: 0 });
    const target = { lat: 64.1801, lon: -51.7406, altitude: 3 };
    const local = frame.toRapier(target);
    assert.ok(Math.abs(local.x) < 100);
    assert.ok(Math.abs(local.z) < 100);
    assert.ok(Math.abs(local.y) < 10);
    const restored = frame.fromRapier(local);
    assert.ok(Math.abs(restored.lat - target.lat) < 1e-8);
    assert.ok(Math.abs(restored.lon - target.lon) < 1e-8);
    assert.ok(Math.abs(restored.altitude - target.altitude) < 0.01);
});

test('runtime database gates reject production and non-test names', () => {
    assert.throws(() => assertTestOnlyRuntime({
        env: { NODE_ENV: 'production' },
        bankDbPath: 'bank.test.sqlite',
        worldDbPath: 'world.test.sqlite'
    }), /refuses NODE_ENV=production/);
    assert.throws(() => assertTestOnlyRuntime({
        env: { DATABASE_URL: 'postgresql://prod/prod' },
        bankDbPath: 'bank.test.sqlite',
        worldDbPath: 'world.test.sqlite'
    }), /DATABASE_URL is forbidden/);
    assert.throws(
        () => validatePostgresTestUrl('postgresql://localhost/greenland_bank'),
        /must end with _test/
    );
    assert.equal(
        validatePostgresTestUrl('postgresql://localhost/greenland_bank_test').databaseName,
        'greenland_bank_test'
    );
});

test('the dedicated API assembles against isolated test SQLite files', async (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'greenland-api-'));
    const env = {
        NODE_ENV: 'test',
        BANK_DB_PATH: path.join(directory, 'bank.test.sqlite'),
        WORLD_DB_PATH: path.join(directory, 'world.test.sqlite')
    };
    const { app, runtime } = createApplication({ env });
    await Promise.all([runtime.bankReady, runtime.worldReady]);
    t.after(async () => {
        await runtime.world.close();
        await runtime.bank.close();
        fs.rmSync(directory, { recursive: true, force: true });
    });

    assert.equal(fs.existsSync(env.BANK_DB_PATH), true);
    assert.equal(fs.existsSync(env.WORLD_DB_PATH), true);
    assert.ok(app._router.stack.some((layer) => layer.route?.path === '/health'));
    assert.ok(app._router.stack.some((layer) => String(layer.regexp).includes('api\\/bank')));
    assert.ok(app._router.stack.some((layer) => String(layer.regexp).includes('api\\/game')));
});

test('Redis configuration is isolated and never becomes bank authority', () => {
    assert.deepEqual(validateRedisTestConfig(), {
        url: 'redis://127.0.0.1:6379/15',
        hostname: '127.0.0.1',
        database: 15,
        namespace: 'greenland:test:',
        local: true
    });
    assert.throws(
        () => validateRedisTestConfig({ url: 'redis://redis.production.example/0' }),
        /Remote Redis requires/
    );
    assert.throws(
        () => validateRedisTestConfig({ namespace: 'greenland:production:' }),
        /must start with greenland:test:/
    );
});

test('Redis bridge fans out rooms, keeps replay, presence, and expiring leases', async () => {
    const { EventEmitter } = require('node:events');
    class FakeRedis extends EventEmitter {
        constructor() {
            super();
            this.calls = [];
            this.replayResponse = null;
            this.leaseResult = 'OK';
        }
        async subscribe(...args) { this.calls.push(['subscribe', ...args]); }
        async unsubscribe(...args) { this.calls.push(['unsubscribe', ...args]); }
        async publish(...args) { this.calls.push(['publish', ...args]); return 1; }
        async xadd(...args) { this.calls.push(['xadd', ...args]); return '7-0'; }
        async xread(...args) { this.calls.push(['xread', ...args]); return this.replayResponse; }
        async set(...args) { this.calls.push(['set', ...args]); return this.leaseResult; }
        async eval(...args) { this.calls.push(['eval', ...args]); return 1; }
    }
    const publisher = new FakeRedis();
    const subscriber = new FakeRedis();
    const bridge = new RedisRealtimeBridge({ publisher, subscriber });
    const envelope = createEnvelope({
        kind: 'snapshot', roomId: 'port:NUUK', sequence: 1, tick: 2, payload: { boats: 1 }
    });
    let received = null;
    await bridge.subscribe('port:NUUK', (message) => { received = message; });
    subscriber.emit('message', bridge.channel('port:NUUK'), JSON.stringify(envelope));
    assert.deepEqual(received, envelope);

    const published = await bridge.publish('port:NUUK', envelope);
    assert.equal(published.streamId, '7-0');
    assert.ok(publisher.calls.some((call) => call[0] === 'xadd'));
    assert.ok(publisher.calls.some((call) => call[0] === 'publish'));
    await assert.rejects(
        () => bridge.publish('port:SISIMIUT', envelope),
        /roomId must match/
    );

    publisher.replayResponse = [[bridge.stream('port:NUUK'), [['7-0', ['message', JSON.stringify(envelope)]]]]];
    assert.equal((await bridge.replay('port:NUUK'))[0].envelope.roomId, 'port:NUUK');
    await bridge.setPresence({ accountId: 'account-1', serverId: 'server-1', roomId: 'port:NUUK' });
    assert.ok(publisher.calls.some((call) => call[0] === 'set' && call.includes('PX')));
    assert.equal(await bridge.acquireLease({ resourceId: 'port:NUUK', holderId: 'server-1' }), true);
    assert.equal(await bridge.releaseLease({ resourceId: 'port:NUUK', holderId: 'server-1' }), true);
    await bridge.stop();
});
