const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { BankService } = require('../packages/bank/src/bank-service');
const { GameError, GameService } = require('../packages/world/src/game-service');

async function fixture(t) {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'greenland-game-'));
    const bank = new BankService({ dbPath: path.join(directory, 'bank.db') });
    await bank.open();
    let now = new Date('2026-07-17T10:00:00.000Z');
    const game = new GameService({
        dbPath: path.join(directory, 'game.db'),
        bank,
        clock: () => new Date(now)
    });
    await game.open();
    t.after(async () => {
        await game.close();
        await bank.close();
        fs.rmSync(directory, { recursive: true, force: true });
    });
    return {
        bank,
        game,
        get now() { return new Date(now); },
        setNow(value) { now = new Date(value); }
    };
}

test('the game server seeds the WebGPU world with explicit Greenland ports and sea routes', async (t) => {
    const { game } = await fixture(t);
    const world = await game.getWorldState();

    assert.equal(world.renderer, 'webgpu');
    assert.deepEqual(world.nodes.map((node) => node.code), ['ILULISSAT', 'NUUK', 'SISIMIUT']);
    assert.equal(world.routes.length, 4);
    assert.ok(world.routes.every((route) => route.mode === 'sea'));
    assert.ok(world.routes.every((route) => route.path.length >= 2));
    assert.equal(world.markets.length, 9);
});

test('an authenticated game identity receives one persistent bank-owned UUID starter boat', async (t) => {
    const { bank, game } = await fixture(t);
    const identity = { externalUserId: 'x_user:42', displayName: 'Captain Ada' };

    const first = await game.bootstrapPlayer(identity);
    const second = await game.bootstrapPlayer({ ...identity, displayName: 'Changed label' });

    assert.equal(first.created, true);
    assert.equal(second.created, false);
    assert.equal(first.player.id, second.player.id);
    assert.equal(first.vehicle.assetId, second.vehicle.assetId);
    assert.equal(first.vehicle.currentNodeCode, 'NUUK');
    assert.equal(first.vehicle.status, 'docked');

    const asset = await bank.getAsset(first.vehicle.assetId);
    assert.equal(asset.kind, 'vehicle');
    assert.equal(asset.assetType, 'coastal_cargo_boat');
    assert.equal(asset.ownerAccountId, first.player.bankAccountId);
});

test('cargo accepts bank-owned resource lots, enforces capacity, and remains UUID-referenced', async (t) => {
    const { bank, game } = await fixture(t);
    const joined = await game.bootstrapPlayer({ externalUserId: 'x_user:51', displayName: 'Naja' });
    await bank.registerParcel({ tileId: '12-1375-791', ownerAccountId: joined.player.bankAccountId });
    const lot = await bank.issueAsset({
        kind: 'resource_lot',
        assetType: 'fish',
        ownerAccountId: joined.player.bankAccountId,
        quantity: 12.5,
        unit: 'tonnes',
        originTileId: '12-1375-791'
    });
    const oversized = await bank.issueAsset({
        kind: 'resource_lot',
        assetType: 'food_supplies',
        ownerAccountId: joined.player.bankAccountId,
        quantity: 9,
        unit: 'tonnes',
        originTileId: '12-1375-791'
    });

    const loaded = await game.loadCargo({
        externalUserId: joined.player.externalUserId,
        vehicleAssetId: joined.vehicle.assetId,
        resourceAssetId: lot.id
    });
    assert.equal(loaded.cargoLoadTonnes, 12.5);
    assert.equal(loaded.cargo[0].resourceAssetId, lot.id);

    await assert.rejects(
        () => game.loadCargo({
            externalUserId: joined.player.externalUserId,
            vehicleAssetId: joined.vehicle.assetId,
            resourceAssetId: oversized.id
        }),
        (error) => error instanceof GameError && error.code === 'CARGO_CAPACITY_EXCEEDED'
    );
});

test('boat voyages are server-timed, idempotent, interpolated, and dock only on arrival', async (t) => {
    const context = await fixture(t);
    const { game } = context;
    const joined = await game.bootstrapPlayer({ externalUserId: 'x_user:84', displayName: 'Aputsiaq' });

    const quote = await game.quoteVoyage({
        externalUserId: joined.player.externalUserId,
        vehicleAssetId: joined.vehicle.assetId,
        destinationNodeCode: 'SISIMIUT',
        now: context.now
    });
    assert.equal(quote.route.id, 'sea:NUUK:SISIMIUT');
    assert.equal(quote.route.durationSeconds, 36000);
    assert.ok(quote.fuelUsedLiters > 0);

    const departed = await game.departVoyage({
        externalUserId: joined.player.externalUserId,
        vehicleAssetId: joined.vehicle.assetId,
        destinationNodeCode: 'SISIMIUT',
        idempotencyKey: 'voyage:x_user:84:001',
        now: context.now
    });
    const retried = await game.departVoyage({
        externalUserId: joined.player.externalUserId,
        vehicleAssetId: joined.vehicle.assetId,
        destinationNodeCode: 'SISIMIUT',
        idempotencyKey: 'voyage:x_user:84:001',
        now: context.now
    });
    assert.equal(retried.id, departed.id);
    assert.equal(departed.status, 'in_transit');
    assert.equal((await game.getVehicle(joined.vehicle.assetId)).currentNodeCode, null);

    context.setNow('2026-07-17T15:00:00.000Z');
    const halfway = await game.getTransit(departed.id, context.now);
    assert.equal(halfway.status, 'in_transit');
    assert.equal(halfway.progress, 0.5);
    assert.ok(halfway.position.lat > 64.17 && halfway.position.lat < 66.94);

    context.setNow('2026-07-17T20:00:01.000Z');
    const arrived = await game.getTransit(departed.id, context.now);
    const vehicle = await game.getVehicle(joined.vehicle.assetId);
    assert.equal(arrived.status, 'arrived');
    assert.equal(arrived.progress, 1);
    assert.equal(vehicle.status, 'docked');
    assert.equal(vehicle.currentNodeCode, 'SISIMIUT');

    const world = await game.getWorldState({ now: context.now });
    assert.equal(world.activeTransits.length, 0);
    assert.ok(world.events.some((event) => event.eventType === 'voyage_departed'));
    assert.ok(world.events.some((event) => event.eventType === 'voyage_arrived'));
});

test('a different authenticated player cannot command another player bank-owned boat', async (t) => {
    const { game } = await fixture(t);
    const owner = await game.bootstrapPlayer({ externalUserId: 'x_user:100', displayName: 'Owner' });
    const attacker = await game.bootstrapPlayer({ externalUserId: 'x_user:101', displayName: 'Attacker' });

    await assert.rejects(
        () => game.quoteVoyage({
            externalUserId: attacker.player.externalUserId,
            vehicleAssetId: owner.vehicle.assetId,
            destinationNodeCode: 'SISIMIUT'
        }),
        (error) => error instanceof GameError && error.code === 'VEHICLE_NOT_CONTROLLED'
    );
});

test('cargo cannot be unloaded while a boat is in transit', async (t) => {
    const { bank, game, now } = await fixture(t);
    const joined = await game.bootstrapPlayer({ externalUserId: 'x_user:110', displayName: 'Sila' });
    await bank.registerParcel({ tileId: '12-1376-791', ownerAccountId: joined.player.bankAccountId });
    const lot = await bank.issueAsset({
        kind: 'resource_lot',
        assetType: 'fish',
        ownerAccountId: joined.player.bankAccountId,
        quantity: 1,
        unit: 'tonnes',
        originTileId: '12-1376-791'
    });
    await game.loadCargo({
        externalUserId: joined.player.externalUserId,
        vehicleAssetId: joined.vehicle.assetId,
        resourceAssetId: lot.id
    });
    await game.departVoyage({
        externalUserId: joined.player.externalUserId,
        vehicleAssetId: joined.vehicle.assetId,
        destinationNodeCode: 'SISIMIUT',
        idempotencyKey: 'voyage:x_user:110:001',
        now
    });

    await assert.rejects(
        () => game.unloadCargo({
            externalUserId: joined.player.externalUserId,
            vehicleAssetId: joined.vehicle.assetId,
            resourceAssetId: lot.id
        }),
        (error) => error instanceof GameError && error.code === 'VEHICLE_NOT_DOCKED'
    );
});
