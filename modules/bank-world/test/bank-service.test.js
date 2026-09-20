const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { v4: uuidv4 } = require('uuid');

const { BankError, BankService } = require('../packages/bank/src/bank-service');
const { validateTestDatabaseUrl } = require('../packages/bank/src/database-safety');

async function fixture(t) {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'greenland-bank-'));
    const bank = new BankService({ dbPath: path.join(directory, 'bank.db') });
    await bank.open();
    t.after(async () => {
        await bank.close();
        fs.rmSync(directory, { recursive: true, force: true });
    });

    const treasury = await bank.createAccount({
        displayName: 'Bank of Greenland Treasury',
        accountType: 'treasury'
    });
    const player = await bank.createAccount({
        externalUserId: 'x_user:42',
        displayName: 'Captain Ada',
        accountType: 'player'
    });
    const buyer = await bank.createAccount({
        externalUserId: 'x_user:84',
        displayName: 'Captain Naja',
        accountType: 'player'
    });
    const warehouse = await bank.createAccount({
        displayName: 'Nuuk Verified Warehouse',
        accountType: 'warehouse'
    });

    return { bank, treasury, player, buyer, warehouse };
}

test('credit retries preserve the transaction UUID and reject changed terms', async (t) => {
    const { bank, treasury, player, buyer } = await fixture(t);
    await bank.transferCredits({ fromAccountId: treasury.id, toAccountId: player.id,
        amount: 100, idempotencyKey: 'fund-credit-retry' });
    const args = { fromAccountId: player.id, toAccountId: buyer.id,
        amount: 10, idempotencyKey: 'credit-retry' };
    const first = await bank.transferCredits(args);
    assert.equal((await bank.transferCredits(args)).id, first.id);
    await assert.rejects(() => bank.transferCredits({ ...args, amount: 20 }),
        error => error.code === 'IDEMPOTENCY_CONFLICT');
    assert.equal((await bank.getBalance(player.id)).balance, 90);
});

test('Terrain source registration is retry-safe and cannot reassign existing ownership', async (t) => {
    const { bank, player, buyer } = await fixture(t);
    const args = { kind: 'vehicle', assetType: 'patria-amv', ownerAccountId: player.id,
        sourceNamespace: 'terrain:nuuk', sourceId: 'amv-01' };
    const first = await bank.issueAsset(args);
    assert.equal((await bank.issueAsset(args)).id, first.id);
    await assert.rejects(() => bank.issueAsset({ ...args, ownerAccountId: buyer.id }),
        error => error.code === 'SOURCE_BINDING_CONFLICT');
    assert.equal((await bank.getPortfolio(player.id)).ownedAssets.length, 1);
});

test('PostgreSQL validation refuses production-looking database targets', () => {
    assert.throws(
        () => validateTestDatabaseUrl('postgresql://bank@db.example.com/greenland_game'),
        /must end with _test/
    );
    assert.throws(
        () => validateTestDatabaseUrl('postgresql://bank@db.example.com/greenland_game_test'),
        /explicit allowRemote/
    );
    assert.deepEqual(
        validateTestDatabaseUrl('postgresql://bank@127.0.0.1:5433/greenland_game_test'),
        {
            hostname: '127.0.0.1',
            port: '5433',
            databaseName: 'greenland_game_test',
            local: true
        }
    );
});

test('a depth-12 terrain parcel is represented by a UUID title asset', async (t) => {
    const { bank, treasury } = await fixture(t);
    const parcel = await bank.registerParcel({
        tileId: '12-1375-791',
        ownerAccountId: treasury.id,
        metadata: { projectedSideM: 659.1796875 }
    });

    assert.match(parcel.titleAssetId, /^[0-9a-f-]{36}$/);
    assert.equal(parcel.ownerAccountId, treasury.id);
    assert.equal(parcel.metadata.projectedSideM, 659.1796875);

    const portfolio = await bank.getPortfolio(treasury.id);
    assert.equal(portfolio.parcels.length, 1);
    assert.equal(portfolio.ownedAssets[0].kind, 'land_title');

    await assert.rejects(
        () => bank.registerParcel({ tileId: '11-687-395', ownerAccountId: treasury.id }),
        (error) => error instanceof BankError && error.code === 'INVALID_PARCEL_DEPTH'
    );
});

test('an Atlantis x_user identity resolves to one separate game-bank account', async (t) => {
    const { bank } = await fixture(t);
    const first = await bank.resolveAccount({
        externalUserId: 'x_user:125',
        displayName: 'Kalaallit Player'
    });
    const second = await bank.resolveAccount({
        externalUserId: 'x_user:125',
        displayName: 'Changed display name is not identity'
    });

    assert.equal(first.id, second.id);
    assert.equal(first.externalUserId, 'x_user:125');
    assert.equal(second.displayName, 'Kalaallit Player');
});

test('a player atomically purchases bank-listed native terrain parcels', async (t) => {
    const { bank, treasury, player } = await fixture(t);
    await bank.registerParcel({
        tileId: '12-1380-791',
        ownerAccountId: treasury.id,
        saleStatus: 'for_sale',
        priceAmount: 250
    });
    await bank.transferCredits({
        fromAccountId: treasury.id,
        toAccountId: player.id,
        amount: 300,
        idempotencyKey: 'starting-funds:land-buyer'
    });
    const purchase = await bank.purchaseParcels({
        tileIds: ['12-1380-791'],
        buyerAccountId: player.id,
        idempotencyKey: 'land-purchase:001'
    });

    assert.equal(purchase.parcels[0].ownerAccountId, player.id);
    assert.equal(purchase.parcels[0].saleStatus, 'owned');
    assert.equal((await bank.getBalance(player.id)).balance, 50);
    assert.equal((await bank.getBalance(treasury.id)).balance, -50);
    const retried = await bank.purchaseParcels({
        tileIds: ['12-1380-791'],
        buyerAccountId: player.id,
        idempotencyKey: 'land-purchase:001'
    });
    assert.equal(retried.transaction.id, purchase.transaction.id);
});

test('resource lots are authentic, traceable, and retain distinct ownership and custody', async (t) => {
    const { bank, treasury, player, buyer, warehouse } = await fixture(t);
    await bank.registerParcel({ tileId: '12-1375-791', ownerAccountId: treasury.id });

    const lot = await bank.issueAsset({
        kind: 'resource_lot',
        assetType: 'rare_earth',
        ownerAccountId: player.id,
        quantity: 18.4,
        unit: 'kg',
        originTileId: '12-1375-791',
        metadata: { producedByAssetId: uuidv4(), productionEventId: uuidv4() }
    });

    const verified = await bank.verifyAsset(lot.id);
    assert.equal(verified.authentic, true);
    assert.equal(verified.spendable, true);

    const deposited = await bank.changeCustody({
        assetId: lot.id,
        ownerAccountId: player.id,
        toCustodianAccountId: warehouse.id
    });
    assert.equal(deposited.ownerAccountId, player.id);
    assert.equal(deposited.custodianAccountId, warehouse.id);
    assert.equal(deposited.status, 'stored');

    const transferred = await bank.transferAsset({
        assetId: lot.id,
        fromOwnerAccountId: player.id,
        toOwnerAccountId: buyer.id
    });
    assert.equal(transferred.ownerAccountId, buyer.id);
    assert.equal(transferred.custodianAccountId, warehouse.id);

    const warehousePortfolio = await bank.getPortfolio(warehouse.id);
    assert.equal(warehousePortfolio.assetsInCustody[0].id, lot.id);
});

test('splitting a resource lot conserves quantity and consumes the parent UUID', async (t) => {
    const { bank, treasury, player } = await fixture(t);
    await bank.registerParcel({ tileId: '12-1376-791', ownerAccountId: treasury.id });
    const lot = await bank.issueAsset({
        kind: 'resource_lot',
        assetType: 'minerals',
        ownerAccountId: player.id,
        quantity: 18.4,
        unit: 'tonnes',
        originTileId: '12-1376-791'
    });

    await assert.rejects(
        () => bank.splitResourceLot({
            assetId: lot.id,
            ownerAccountId: player.id,
            quantities: [10, 9]
        }),
        (error) => error instanceof BankError && error.code === 'QUANTITY_NOT_CONSERVED'
    );

    const split = await bank.splitResourceLot({
        assetId: lot.id,
        ownerAccountId: player.id,
        quantities: [10, 8.4]
    });
    assert.equal(split.parent.status, 'consumed');
    assert.deepEqual(split.children.map((child) => child.quantity), [10, 8.4]);

    const parentVerification = await bank.verifyAsset(lot.id);
    assert.equal(parentVerification.authentic, true);
    assert.equal(parentVerification.spendable, false);

    const provenance = await bank.getProvenance(split.children[0].id);
    assert.equal(provenance.lineage[0].parentAssetId, lot.id);
    assert.ok(provenance.events.some((event) => event.eventType === 'resource_issued'));
    assert.ok(provenance.events.some((event) => event.eventType === 'resource_created_from_split'));
});

test('validated production and recipes preserve UUID provenance from mine to output', async (t) => {
    const { bank, player } = await fixture(t);
    const tileId = '12-1381-791';
    await bank.registerParcel({ tileId, ownerAccountId: player.id });
    const mine = await bank.issueAsset({
        kind: 'machinery',
        assetType: 'rare_earth_mine',
        ownerAccountId: player.id,
        locationTileId: tileId
    });
    const refinery = await bank.issueAsset({
        kind: 'machinery',
        assetType: 'ore_refinery',
        ownerAccountId: player.id,
        locationTileId: tileId
    });
    const rule = await bank.registerProductionRule({
        producerAssetType: 'rare_earth_mine',
        outputAssetType: 'rare_earth_ore',
        outputUnit: 'tonnes',
        quantityPerHour: 10
    });
    const production = await bank.claimProduction({
        ruleId: rule.id,
        producerAssetId: mine.id,
        ownerAccountId: player.id,
        originTileId: tileId,
        elapsedSeconds: 3600,
        sourceEventId: 'simulation:mine:tick-7200',
        idempotencyKey: 'production:mine:hour-1',
        evidence: { simulationTick: 7200 }
    });
    assert.equal(production.output.quantity, 10);
    assert.equal(production.output.metadata.producerAssetId, mine.id);
    await assert.rejects(
        () => bank.claimProduction({
            ruleId: rule.id,
            producerAssetId: mine.id,
            ownerAccountId: player.id,
            originTileId: tileId,
            elapsedSeconds: 3600,
            sourceEventId: 'simulation:mine:tick-7200',
            idempotencyKey: 'production:mine:replay-with-new-key'
        }),
        (error) => error instanceof BankError && error.code === 'SOURCE_EVENT_REPLAY'
    );

    const recipe = await bank.registerRecipe({
        recipeName: 'refine-rare-earth-ore',
        processorAssetType: 'ore_refinery',
        inputs: [{ assetType: 'rare_earth_ore', quantity: 10, unit: 'tonnes' }],
        outputs: [{ assetType: 'rare_earth_concentrate', quantity: 5, unit: 'tonnes' }]
    });
    const transformed = await bank.transformResources({
        recipeId: recipe.id,
        processorAssetId: refinery.id,
        ownerAccountId: player.id,
        originTileId: tileId,
        inputAssetIds: [production.output.id],
        sourceEventId: 'simulation:refinery:tick-7201',
        idempotencyKey: 'transform:refinery:batch-1'
    });
    assert.equal(transformed.inputs[0].status, 'consumed');
    assert.equal(transformed.outputs[0].quantity, 5);
    assert.equal(transformed.outputs[0].assetType, 'rare_earth_concentrate');

    const provenance = await bank.getProvenance(transformed.outputs[0].id);
    assert.ok(provenance.assets.some((asset) => asset.id === production.output.id));
    assert.ok(provenance.events.some((event) => event.eventType === 'resource_produced'));
    assert.ok(provenance.events.some((event) => event.eventType === 'resource_created_by_recipe'));
});

test('ownership checks prevent unauthorized transfers', async (t) => {
    const { bank, treasury, player, buyer } = await fixture(t);
    await bank.registerParcel({ tileId: '12-1377-791', ownerAccountId: treasury.id });
    const vehicle = await bank.issueAsset({
        kind: 'vehicle',
        assetType: 'snowmobile',
        ownerAccountId: player.id
    });

    await assert.rejects(
        () => bank.transferAsset({
            assetId: vehicle.id,
            fromOwnerAccountId: buyer.id,
            toOwnerAccountId: treasury.id
        }),
        (error) => error instanceof BankError && error.code === 'OWNER_MISMATCH'
    );
});

test('the central bank atomically settles warehouse payment, commission, and ownership', async (t) => {
    const { bank, treasury, player, buyer, warehouse } = await fixture(t);
    await bank.registerParcel({ tileId: '12-1378-791', ownerAccountId: treasury.id });
    await bank.transferCredits({
        fromAccountId: treasury.id,
        toAccountId: buyer.id,
        amount: 1000,
        idempotencyKey: 'starting-funds:buyer'
    });
    const lot = await bank.issueAsset({
        kind: 'resource_lot',
        assetType: 'fish',
        ownerAccountId: player.id,
        quantity: 20,
        unit: 'crates',
        originTileId: '12-1378-791'
    });
    await bank.changeCustody({
        assetId: lot.id,
        ownerAccountId: player.id,
        toCustodianAccountId: warehouse.id
    });

    const service = await bank.registerServiceIdentity({
        serviceName: 'nuuk-warehouse',
        serviceType: 'warehouse',
        ownerAccountId: warehouse.id
    });
    assert.equal((await bank.authenticateServiceToken(service.token)).ownerAccountId, warehouse.id);
    await bank.createWarehouseListing({
        assetId: lot.id,
        sellerAccountId: player.id,
        warehouseAccountId: warehouse.id,
        minimumGrossAmount: 90,
        idempotencyKey: 'listing:quote-001'
    });
    const quote = await bank.createWarehouseQuote({
        assetIds: [lot.id],
        sellerAccountId: player.id,
        warehouseAccountId: warehouse.id,
        grossAmount: 100,
        commissionAmount: 5,
        clientQuoteId: 'quote-001'
    });
    const request = {
        quoteId: quote.id,
        buyerAccountId: buyer.id,
        actorAccountId: treasury.id,
        idempotencyKey: 'warehouse:quote-001'
    };
    const settled = await bank.settleWarehouseQuote(request);

    assert.equal(settled.transaction.transactionType, 'warehouse_trade');
    assert.equal(settled.transaction.status, 'posted');
    assert.equal(
        settled.transaction.entries.reduce((sum, entry) => sum + entry.amount, 0),
        0
    );
    assert.equal(settled.assets[0].ownerAccountId, buyer.id);
    assert.equal(settled.assets[0].custodianAccountId, warehouse.id);
    assert.equal((await bank.getBalance(buyer.id)).balance, 900);
    assert.equal((await bank.getBalance(player.id)).balance, 95);
    assert.equal((await bank.getBalance(warehouse.id)).balance, 5);

    const retried = await bank.settleWarehouseQuote(request);
    assert.equal(retried.transaction.id, settled.transaction.id);
    assert.equal((await bank.getBalance(buyer.id)).balance, 900);

    const provenance = await bank.getProvenance(lot.id);
    assert.ok(provenance.events.some((event) => (
        event.eventType === 'warehouse_trade_settled'
        && event.transactionId === settled.transaction.id
    )));
});

test('a partial warehouse sale atomically turns 12.5 tonnes into sold and retained UUID lots', async (t) => {
    const { bank, treasury, player, buyer, warehouse } = await fixture(t);
    await bank.registerParcel({ tileId: '12-1384-791', ownerAccountId: treasury.id });
    await bank.transferCredits({
        fromAccountId: treasury.id,
        toAccountId: buyer.id,
        amount: 500,
        idempotencyKey: 'starting-funds:partial-buyer'
    });
    const parent = await bank.issueAsset({
        kind: 'resource_lot',
        assetType: 'fish',
        ownerAccountId: player.id,
        quantity: 12.5,
        unit: 'tonnes',
        originTileId: '12-1384-791'
    });
    await bank.changeCustody({
        assetId: parent.id,
        ownerAccountId: player.id,
        toCustodianAccountId: warehouse.id
    });
    await bank.createWarehouseListing({
        assetId: parent.id,
        sellerAccountId: player.id,
        warehouseAccountId: warehouse.id,
        minimumGrossAmount: 250,
        idempotencyKey: 'listing:partial-fish'
    });
    const quote = await bank.createWarehouseQuote({
        assetIds: [parent.id],
        resourceLines: [{ assetId: parent.id, quantity: 5 }],
        sellerAccountId: player.id,
        warehouseAccountId: warehouse.id,
        grossAmount: 120,
        commissionAmount: 6,
        clientQuoteId: 'quote:partial-fish'
    });
    const request = {
        quoteId: quote.id,
        buyerAccountId: buyer.id,
        actorAccountId: treasury.id,
        idempotencyKey: 'settle:partial-fish'
    };
    const settled = await bank.settleWarehouseQuote(request);

    assert.equal(settled.transaction.transactionType, 'warehouse_partial_trade');
    assert.equal(settled.assets.length, 1);
    assert.equal(settled.assets[0].quantity, 5);
    assert.equal(settled.assets[0].ownerAccountId, buyer.id);
    assert.equal(settled.assets[0].custodianAccountId, warehouse.id);
    assert.equal(settled.remainderAssets.length, 1);
    assert.equal(settled.remainderAssets[0].quantity, 7.5);
    assert.equal(settled.remainderAssets[0].ownerAccountId, player.id);
    assert.equal((await bank.getAsset(parent.id)).status, 'consumed');
    assert.equal((await bank.getBalance(buyer.id)).balance, 380);
    assert.equal((await bank.getBalance(player.id)).balance, 114);
    assert.equal((await bank.getBalance(warehouse.id)).balance, 6);

    const provenance = await bank.getProvenance(settled.assets[0].id);
    assert.ok(provenance.lineage.some((edge) => edge.parentAssetId === parent.id));
    assert.ok(provenance.events.some((event) => event.eventType === 'resource_partially_sold'));
    assert.ok(provenance.events.some((event) => event.eventType === 'resource_created_for_partial_sale'));

    const retried = await bank.settleWarehouseQuote(request);
    assert.equal(retried.transaction.id, settled.transaction.id);
    assert.equal(retried.assets[0].id, settled.assets[0].id);
    assert.equal(retried.remainderAssets[0].id, settled.remainderAssets[0].id);

    const secondQuote = await bank.createWarehouseQuote({
        assetIds: [settled.remainderAssets[0].id],
        resourceLines: [{ assetId: settled.remainderAssets[0].id, quantity: 1 }],
        sellerAccountId: player.id,
        warehouseAccountId: warehouse.id,
        grossAmount: 25,
        commissionAmount: 1,
        clientQuoteId: 'quote:partial-fish:second'
    });
    await assert.rejects(
        () => bank.settleWarehouseQuote({ ...request, quoteId: secondQuote.id }),
        (error) => error instanceof BankError && error.code === 'IDEMPOTENCY_CONFLICT'
    );
});

test('warehouse settlement rolls back completely when the buyer lacks funds', async (t) => {
    const { bank, treasury, player, buyer, warehouse } = await fixture(t);
    await bank.registerParcel({ tileId: '12-1379-791', ownerAccountId: treasury.id });
    const lot = await bank.issueAsset({
        kind: 'resource_lot',
        assetType: 'medical',
        ownerAccountId: player.id,
        quantity: 2,
        unit: 'kits',
        originTileId: '12-1379-791'
    });
    await bank.changeCustody({
        assetId: lot.id,
        ownerAccountId: player.id,
        toCustodianAccountId: warehouse.id
    });

    await assert.rejects(
        () => bank.createWarehouseQuote({
            assetIds: [lot.id],
            sellerAccountId: player.id,
            warehouseAccountId: warehouse.id,
            grossAmount: 100,
            commissionAmount: 5,
            clientQuoteId: 'without-seller-consent'
        }),
        (error) => error instanceof BankError && error.code === 'SALE_LISTING_REQUIRED'
    );
    await bank.createWarehouseListing({
        assetId: lot.id,
        sellerAccountId: player.id,
        warehouseAccountId: warehouse.id,
        minimumGrossAmount: 100,
        idempotencyKey: 'listing:no-funds'
    });

    const quote = await bank.createWarehouseQuote({
        assetIds: [lot.id],
        sellerAccountId: player.id,
        warehouseAccountId: warehouse.id,
        grossAmount: 100,
        commissionAmount: 5,
        clientQuoteId: 'no-funds'
    });
    await assert.rejects(
        () => bank.settleWarehouseQuote({
            quoteId: quote.id,
            buyerAccountId: buyer.id,
            actorAccountId: treasury.id,
            idempotencyKey: 'warehouse:no-funds'
        }),
        (error) => error instanceof BankError && error.code === 'INSUFFICIENT_FUNDS'
    );

    const unchanged = await bank.getAsset(lot.id);
    assert.equal(unchanged.ownerAccountId, player.id);
    assert.equal(unchanged.custodianAccountId, warehouse.id);
    assert.equal((await bank.getBalance(player.id)).balance, 0);
    assert.equal((await bank.getBalance(warehouse.id)).balance, 0);
    assert.equal((await bank.getWarehouseQuote(quote.id)).status, 'open');
});

test('warehouse credentials are scoped to the registered warehouse account', async (t) => {
    const { bank, player, warehouse } = await fixture(t);
    await assert.rejects(
        () => bank.registerServiceIdentity({
            serviceName: 'fake-warehouse',
            serviceType: 'warehouse',
            ownerAccountId: player.id
        }),
        (error) => error instanceof BankError && error.code === 'NOT_A_WAREHOUSE'
    );
    const identity = await bank.registerServiceIdentity({
        serviceName: 'nuuk-warehouse-2',
        serviceType: 'warehouse',
        ownerAccountId: warehouse.id
    });
    assert.match(identity.token, /^gbs_/);
    await assert.rejects(
        () => bank.authenticateServiceToken(`${identity.token}-tampered`),
        (error) => error instanceof BankError && error.code === 'SERVICE_AUTH_REQUIRED'
    );
});
