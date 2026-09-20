const sqlite3 = require('sqlite3').verbose();
const crypto = require('crypto');
const { v4: uuidv4, validate: uuidValidate } = require('uuid');

const TILE_ID_PATTERN = /^(\d+)-(\d+)-(\d+)$/;
const ACTIVE_ASSET_STATES = new Set(['active', 'stored', 'in_transit', 'installed', 'deployed']);
const ACCOUNT_TYPES = new Set(['player', 'warehouse', 'treasury', 'system']);
const SERVICE_TYPES = new Set(['warehouse', 'game_authority']);
const ASSET_KINDS = new Set([
    'resource_lot',
    'vehicle',
    'equipment',
    'machinery',
    'powerplant',
    'warehouse',
    'structure',
    'land_title'
]);

class BankError extends Error {
    constructor(message, code = 'BANK_ERROR', status = 400) {
        super(message);
        this.name = 'BankError';
        this.code = code;
        this.status = status;
    }
}

function parseJson(value) {
    if (!value) return {};
    try {
        return JSON.parse(value);
    } catch (_error) {
        return {};
    }
}

function hashServiceToken(token) {
    return crypto.createHash('sha256').update(token).digest('hex');
}

function mapServiceIdentity(row) {
    if (!row) return null;
    return {
        id: row.id,
        serviceName: row.service_name,
        serviceType: row.service_type,
        ownerAccountId: row.owner_account_id,
        ownerName: row.owner_name,
        active: Boolean(row.is_active),
        createdAt: row.created_at
    };
}

function mapWarehouseQuote(row) {
    if (!row) return null;
    const expired = row.status === 'open' && Date.parse(row.expires_at) <= Date.now();
    return {
        id: row.id,
        clientQuoteId: row.client_quote_id,
        warehouseAccountId: row.warehouse_account_id,
        sellerAccountId: row.seller_account_id,
        assetIds: parseJson(row.asset_ids_json),
        listingIds: parseJson(row.listing_ids_json),
        grossAmount: row.gross_amount,
        commissionAmount: row.commission_amount,
        currency: row.currency,
        status: expired ? 'expired' : row.status,
        expiresAt: row.expires_at,
        transactionId: row.transaction_id,
        terms: parseJson(row.terms_json),
        createdAt: row.created_at
    };
}

function mapWarehouseListing(row) {
    if (!row) return null;
    const expired = row.status === 'open' && Date.parse(row.expires_at) <= Date.now();
    return {
        id: row.id,
        assetId: row.asset_id,
        sellerAccountId: row.seller_account_id,
        warehouseAccountId: row.warehouse_account_id,
        minimumGrossAmount: row.minimum_gross_amount,
        currency: row.currency,
        status: expired ? 'expired' : row.status,
        expiresAt: row.expires_at,
        transactionId: row.transaction_id,
        createdAt: row.created_at
    };
}

function requireString(value, name) {
    if (typeof value !== 'string' || !value.trim()) {
        throw new BankError(`${name} is required`, 'INVALID_ARGUMENT');
    }
    return value.trim();
}

function requireUuid(value, name) {
    const normalized = requireString(value, name);
    if (!uuidValidate(normalized)) {
        throw new BankError(`${name} must be a UUID`, 'INVALID_UUID');
    }
    return normalized;
}

function normalizeRecipeMaterials(materials, name) {
    if (!Array.isArray(materials) || materials.length === 0 || materials.length > 25) {
        throw new BankError(`${name} must contain between 1 and 25 materials`, 'INVALID_RECIPE');
    }
    return materials.map((material) => {
        if (!material || typeof material !== 'object') {
            throw new BankError(`${name} entries must be objects`, 'INVALID_RECIPE');
        }
        const assetType = requireString(material.assetType, `${name}.assetType`);
        const unit = requireString(material.unit, `${name}.unit`);
        const quantity = Number(material.quantity);
        if (!Number.isFinite(quantity) || quantity <= 0) {
            throw new BankError(`${name}.quantity must be positive`, 'INVALID_RECIPE');
        }
        return { assetType, unit, quantity };
    });
}

function parseTileId(tileId, requiredDepth = 12) {
    const normalized = requireString(tileId, 'tileId');
    const match = normalized.match(TILE_ID_PATTERN);
    if (!match) {
        throw new BankError('tileId must use depth-col-row format', 'INVALID_TILE_ID');
    }
    const depth = Number(match[1]);
    const col = Number(match[2]);
    const row = Number(match[3]);
    if (depth !== requiredDepth) {
        throw new BankError(
            `Only depth-${requiredDepth} terrain tiles can be registered as parcels`,
            'INVALID_PARCEL_DEPTH'
        );
    }
    return { tileId: `${depth}-${col}-${row}`, depth, col, row };
}

function mapAsset(row) {
    if (!row) return null;
    return {
        id: row.id,
        kind: row.kind,
        assetType: row.asset_type,
        ownerAccountId: row.owner_account_id,
        ownerName: row.owner_name,
        custodianAccountId: row.custodian_account_id,
        custodianName: row.custodian_name,
        quantity: row.quantity,
        unit: row.unit,
        status: row.status,
        originTileId: row.origin_tile_id,
        locationTileId: row.location_tile_id,
        metadata: parseJson(row.metadata_json),
        version: row.version,
        createdAt: row.created_at,
        updatedAt: row.updated_at
    };
}

class BankService {
    constructor({ dbPath = 'bank_of_greenland.db' } = {}) {
        this.dbPath = dbPath;
        this.db = null;
        this.writeQueue = Promise.resolve();
    }

    async open() {
        if (this.db) return this;
        this.db = await new Promise((resolve, reject) => {
            const db = new sqlite3.Database(this.dbPath, (error) => {
                if (error) reject(error);
                else resolve(db);
            });
        });
        await this._exec('PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL; PRAGMA synchronous = FULL; PRAGMA busy_timeout = 5000;');
        await this._initSchema();
        return this;
    }

    async close() {
        if (!this.db) return;
        const db = this.db;
        this.db = null;
        await new Promise((resolve, reject) => db.close((error) => error ? reject(error) : resolve()));
    }

    _exec(sql) {
        return new Promise((resolve, reject) => {
            this.db.exec(sql, (error) => error ? reject(error) : resolve());
        });
    }

    _run(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.run(sql, params, function onRun(error) {
                if (error) reject(error);
                else resolve({ changes: this.changes, lastID: this.lastID });
            });
        });
    }

    _get(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.get(sql, params, (error, row) => error ? reject(error) : resolve(row));
        });
    }

    _all(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.all(sql, params, (error, rows) => error ? reject(error) : resolve(rows));
        });
    }

    _write(work) {
        const operation = this.writeQueue.then(async () => {
            await this._run('BEGIN IMMEDIATE');
            try {
                const result = await work();
                await this._run('COMMIT');
                return result;
            } catch (error) {
                await this._run('ROLLBACK').catch(() => {});
                throw error;
            }
        });
        this.writeQueue = operation.catch(() => {});
        return operation;
    }

    async _initSchema() {
        await this._exec(`CREATE TABLE IF NOT EXISTS bank_source_binding (
            namespace TEXT NOT NULL, source_id TEXT NOT NULL,
            asset_id TEXT NOT NULL UNIQUE REFERENCES bank_asset(id),
            PRIMARY KEY(namespace, source_id)
        );`);
        await this._exec(`
            CREATE TABLE IF NOT EXISTS bank_account (
                id TEXT PRIMARY KEY,
                external_user_id TEXT UNIQUE,
                display_name TEXT NOT NULL,
                account_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bank_asset (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                owner_account_id TEXT NOT NULL REFERENCES bank_account(id),
                custodian_account_id TEXT REFERENCES bank_account(id),
                quantity REAL,
                unit TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                origin_tile_id TEXT,
                location_tile_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (quantity IS NULL OR quantity > 0)
            );

            CREATE TABLE IF NOT EXISTS bank_parcel (
                tile_id TEXT PRIMARY KEY,
                depth INTEGER NOT NULL,
                col INTEGER NOT NULL,
                row INTEGER NOT NULL,
                title_asset_id TEXT NOT NULL UNIQUE REFERENCES bank_asset(id),
                sale_status TEXT NOT NULL DEFAULT 'held',
                price_amount REAL,
                price_currency TEXT NOT NULL DEFAULT 'GLC',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bank_transaction (
                id TEXT PRIMARY KEY,
                transaction_type TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                actor_account_id TEXT REFERENCES bank_account(id),
                status TEXT NOT NULL DEFAULT 'posted',
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bank_ledger_entry (
                id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL REFERENCES bank_transaction(id),
                account_id TEXT NOT NULL REFERENCES bank_account(id),
                currency TEXT NOT NULL DEFAULT 'GLC',
                amount REAL NOT NULL,
                entry_role TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (amount <> 0)
            );

            CREATE TABLE IF NOT EXISTS bank_service_identity (
                id TEXT PRIMARY KEY,
                service_name TEXT NOT NULL UNIQUE,
                service_type TEXT NOT NULL,
                owner_account_id TEXT NOT NULL REFERENCES bank_account(id),
                token_hash TEXT NOT NULL UNIQUE,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bank_warehouse_quote (
                id TEXT PRIMARY KEY,
                client_quote_id TEXT NOT NULL,
                warehouse_account_id TEXT NOT NULL REFERENCES bank_account(id),
                seller_account_id TEXT NOT NULL REFERENCES bank_account(id),
                asset_ids_json TEXT NOT NULL,
                listing_ids_json TEXT NOT NULL,
                gross_amount REAL NOT NULL,
                commission_amount REAL NOT NULL DEFAULT 0,
                currency TEXT NOT NULL DEFAULT 'GLC',
                status TEXT NOT NULL DEFAULT 'open',
                expires_at TEXT NOT NULL,
                transaction_id TEXT REFERENCES bank_transaction(id),
                terms_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (warehouse_account_id, client_quote_id),
                CHECK (gross_amount > 0),
                CHECK (commission_amount >= 0 AND commission_amount <= gross_amount)
            );

            CREATE TABLE IF NOT EXISTS bank_warehouse_listing (
                id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                asset_id TEXT NOT NULL REFERENCES bank_asset(id),
                seller_account_id TEXT NOT NULL REFERENCES bank_account(id),
                warehouse_account_id TEXT NOT NULL REFERENCES bank_account(id),
                minimum_gross_amount REAL NOT NULL DEFAULT 0,
                currency TEXT NOT NULL DEFAULT 'GLC',
                status TEXT NOT NULL DEFAULT 'open',
                expires_at TEXT NOT NULL,
                transaction_id TEXT REFERENCES bank_transaction(id),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (minimum_gross_amount >= 0)
            );

            CREATE TABLE IF NOT EXISTS bank_asset_event (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL REFERENCES bank_asset(id),
                transaction_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor_account_id TEXT REFERENCES bank_account(id),
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bank_asset_lineage (
                parent_asset_id TEXT NOT NULL REFERENCES bank_asset(id),
                child_asset_id TEXT NOT NULL REFERENCES bank_asset(id),
                quantity REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (parent_asset_id, child_asset_id)
            );

            CREATE TABLE IF NOT EXISTS bank_production_rule (
                id TEXT PRIMARY KEY,
                producer_asset_type TEXT NOT NULL,
                output_asset_type TEXT NOT NULL,
                output_unit TEXT NOT NULL,
                quantity_per_hour REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (quantity_per_hour > 0)
            );

            CREATE TABLE IF NOT EXISTS bank_production_claim (
                id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL UNIQUE REFERENCES bank_transaction(id),
                idempotency_key TEXT NOT NULL UNIQUE,
                source_event_id TEXT NOT NULL UNIQUE,
                rule_id TEXT NOT NULL REFERENCES bank_production_rule(id),
                producer_asset_id TEXT NOT NULL REFERENCES bank_asset(id),
                owner_account_id TEXT NOT NULL REFERENCES bank_account(id),
                origin_tile_id TEXT NOT NULL REFERENCES bank_parcel(tile_id),
                elapsed_seconds INTEGER NOT NULL,
                output_asset_id TEXT NOT NULL UNIQUE REFERENCES bank_asset(id),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bank_recipe (
                id TEXT PRIMARY KEY,
                recipe_name TEXT NOT NULL UNIQUE,
                processor_asset_type TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                outputs_json TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bank_transformation (
                id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL UNIQUE REFERENCES bank_transaction(id),
                idempotency_key TEXT NOT NULL UNIQUE,
                source_event_id TEXT NOT NULL UNIQUE,
                recipe_id TEXT NOT NULL REFERENCES bank_recipe(id),
                processor_asset_id TEXT NOT NULL REFERENCES bank_asset(id),
                owner_account_id TEXT NOT NULL REFERENCES bank_account(id),
                origin_tile_id TEXT NOT NULL REFERENCES bank_parcel(tile_id),
                batches REAL NOT NULL,
                input_asset_ids_json TEXT NOT NULL,
                output_asset_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (batches > 0)
            );

            CREATE INDEX IF NOT EXISTS bank_asset_owner_idx ON bank_asset(owner_account_id);
            CREATE INDEX IF NOT EXISTS bank_asset_custodian_idx ON bank_asset(custodian_account_id);
            CREATE INDEX IF NOT EXISTS bank_asset_origin_idx ON bank_asset(origin_tile_id);
            CREATE INDEX IF NOT EXISTS bank_event_asset_idx ON bank_asset_event(asset_id, created_at);
            CREATE INDEX IF NOT EXISTS bank_lineage_child_idx ON bank_asset_lineage(child_asset_id);
            CREATE INDEX IF NOT EXISTS bank_ledger_account_idx ON bank_ledger_entry(account_id, currency);
            CREATE INDEX IF NOT EXISTS bank_ledger_transaction_idx ON bank_ledger_entry(transaction_id);
            CREATE INDEX IF NOT EXISTS bank_service_token_idx ON bank_service_identity(token_hash);
            CREATE INDEX IF NOT EXISTS bank_quote_status_idx ON bank_warehouse_quote(status, expires_at);
            CREATE INDEX IF NOT EXISTS bank_listing_asset_idx ON bank_warehouse_listing(asset_id, status, expires_at);
        `);
        await this._ensureColumn('bank_asset', 'location_tile_id', 'TEXT');
        await this._run('CREATE INDEX IF NOT EXISTS bank_asset_location_idx ON bank_asset(location_tile_id)');
        await this._ensureColumn('bank_parcel', 'price_amount', 'REAL');
        await this._ensureColumn('bank_parcel', 'price_currency', "TEXT NOT NULL DEFAULT 'GLC'");
        await this._ensureColumn('bank_warehouse_quote', 'listing_ids_json', "TEXT NOT NULL DEFAULT '[]'");
        await this._ensureColumn('bank_production_claim', 'source_event_id', 'TEXT');
        await this._ensureColumn('bank_transformation', 'source_event_id', 'TEXT');
        await this._run(
            'CREATE UNIQUE INDEX IF NOT EXISTS bank_production_source_event_idx ON bank_production_claim(source_event_id)'
        );
        await this._run(
            'CREATE UNIQUE INDEX IF NOT EXISTS bank_transformation_source_event_idx ON bank_transformation(source_event_id)'
        );
    }

    async _ensureColumn(tableName, columnName, definition) {
        const columns = await this._all(`PRAGMA table_info(${tableName})`);
        if (!columns.some((column) => column.name === columnName)) {
            await this._run(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition}`);
        }
    }

    async registerServiceIdentity({ serviceName, serviceType, ownerAccountId }) {
        const name = requireString(serviceName, 'serviceName');
        const type = requireString(serviceType, 'serviceType');
        if (!SERVICE_TYPES.has(type)) {
            throw new BankError(`Unsupported serviceType: ${type}`, 'INVALID_SERVICE_TYPE');
        }
        const ownerId = await this._requireAccount(ownerAccountId, 'ownerAccountId');
        const owner = await this._get('SELECT account_type FROM bank_account WHERE id = ?', [ownerId]);
        if (type === 'warehouse' && owner.account_type !== 'warehouse') {
            throw new BankError('A warehouse service must belong to a warehouse account', 'NOT_A_WAREHOUSE', 409);
        }
        const token = `gbs_${crypto.randomBytes(32).toString('base64url')}`;
        const identityId = uuidv4();

        return this._write(async () => {
            try {
                await this._run(
                    `INSERT INTO bank_service_identity
                     (id, service_name, service_type, owner_account_id, token_hash)
                     VALUES (?, ?, ?, ?, ?)`,
                    [identityId, name, type, ownerId, hashServiceToken(token)]
                );
            } catch (error) {
                if (String(error.message).includes('UNIQUE constraint failed')) {
                    throw new BankError('Service name is already registered', 'SERVICE_EXISTS', 409);
                }
                throw error;
            }
            const identity = await this.getServiceIdentity(identityId);
            return { ...identity, token };
        });
    }

    async getServiceIdentity(serviceIdentityId) {
        const id = requireUuid(serviceIdentityId, 'serviceIdentityId');
        const row = await this._get(
            `SELECT s.*, a.display_name AS owner_name
             FROM bank_service_identity s
             JOIN bank_account a ON a.id = s.owner_account_id
             WHERE s.id = ?`,
            [id]
        );
        if (!row) throw new BankError('Service identity not found', 'SERVICE_NOT_FOUND', 404);
        return mapServiceIdentity(row);
    }

    async authenticateServiceToken(token) {
        const normalized = requireString(token, 'serviceToken');
        const row = await this._get(
            `SELECT s.*, a.display_name AS owner_name
             FROM bank_service_identity s
             JOIN bank_account a ON a.id = s.owner_account_id
             WHERE s.token_hash = ? AND s.is_active = 1`,
            [hashServiceToken(normalized)]
        );
        if (!row) throw new BankError('Invalid or inactive service token', 'SERVICE_AUTH_REQUIRED', 401);
        return mapServiceIdentity(row);
    }

    async createWarehouseQuote({
        warehouseAccountId,
        sellerAccountId,
        assetIds,
        resourceLines = null,
        grossAmount,
        commissionAmount = 0,
        currency = 'GLC',
        expiresInSeconds = 300,
        clientQuoteId = uuidv4(),
        terms = {}
    }) {
        const warehouseId = await this._requireAccount(warehouseAccountId, 'warehouseAccountId');
        const sellerId = await this._requireAccount(sellerAccountId, 'sellerAccountId');
        const warehouse = await this._get('SELECT account_type FROM bank_account WHERE id = ?', [warehouseId]);
        if (warehouse.account_type !== 'warehouse') {
            throw new BankError('warehouseAccountId is not a registered warehouse', 'NOT_A_WAREHOUSE', 409);
        }
        if (!Array.isArray(assetIds) || assetIds.length === 0 || assetIds.length > 100) {
            throw new BankError('assetIds must contain between 1 and 100 UUIDs', 'INVALID_ASSET_LIST');
        }
        const normalizedAssetIds = [...new Set(assetIds.map((id) => requireUuid(id, 'assetId')))];
        if (normalizedAssetIds.length !== assetIds.length) {
            throw new BankError('assetIds cannot contain duplicates', 'DUPLICATE_ASSET');
        }
        let normalizedResourceLines = null;
        if (resourceLines != null) {
            if (!Array.isArray(resourceLines) || resourceLines.length !== normalizedAssetIds.length) {
                throw new BankError(
                    'resourceLines must contain one quantity for every quoted asset',
                    'INVALID_RESOURCE_LINES'
                );
            }
            normalizedResourceLines = resourceLines.map((line) => {
                if (!line || typeof line !== 'object') {
                    throw new BankError('resourceLines entries must be objects', 'INVALID_RESOURCE_LINES');
                }
                const assetId = requireUuid(line.assetId, 'resourceLines.assetId');
                const quantity = Number(line.quantity);
                if (!Number.isFinite(quantity) || quantity <= 0) {
                    throw new BankError('resourceLines.quantity must be positive', 'INVALID_QUANTITY');
                }
                return { assetId, quantity };
            });
            const lineIds = normalizedResourceLines.map((line) => line.assetId);
            if (new Set(lineIds).size !== lineIds.length
                || !normalizedAssetIds.every((assetId) => lineIds.includes(assetId))) {
                throw new BankError(
                    'resourceLines must match assetIds exactly without duplicates',
                    'INVALID_RESOURCE_LINES'
                );
            }
        }
        if (!Number.isFinite(grossAmount) || grossAmount <= 0) {
            throw new BankError('grossAmount must be positive', 'INVALID_AMOUNT');
        }
        if (!Number.isFinite(commissionAmount) || commissionAmount < 0 || commissionAmount > grossAmount) {
            throw new BankError('commissionAmount must be between zero and grossAmount', 'INVALID_COMMISSION');
        }
        if (!Number.isInteger(expiresInSeconds) || expiresInSeconds < 5 || expiresInSeconds > 3600) {
            throw new BankError('expiresInSeconds must be an integer from 5 to 3600', 'INVALID_QUOTE_EXPIRY');
        }
        const quoteId = uuidv4();
        const clientId = requireString(clientQuoteId, 'clientQuoteId');
        const normalizedCurrency = requireString(currency, 'currency').toUpperCase();
        const expiresAt = new Date(Date.now() + expiresInSeconds * 1000).toISOString();

        return this._write(async () => {
            let minimumGrossTotal = 0;
            const listingIds = [];
            for (const assetId of normalizedAssetIds) {
                const asset = await this._get('SELECT * FROM bank_asset WHERE id = ?', [assetId]);
                if (!asset) throw new BankError(`Asset not found: ${assetId}`, 'ASSET_NOT_FOUND', 404);
                if (asset.owner_account_id !== sellerId) {
                    throw new BankError(`Seller does not own asset ${assetId}`, 'OWNER_MISMATCH', 409);
                }
                if (asset.custodian_account_id !== warehouseId || asset.status !== 'stored') {
                    throw new BankError(`Asset ${assetId} is not deposited at this warehouse`, 'WAREHOUSE_CUSTODY_REQUIRED', 409);
                }
                const listing = await this._get(
                    `SELECT * FROM bank_warehouse_listing
                     WHERE asset_id = ? AND seller_account_id = ? AND warehouse_account_id = ?
                       AND status = 'open' AND expires_at > ?
                     ORDER BY created_at DESC LIMIT 1`,
                    [assetId, sellerId, warehouseId, new Date().toISOString()]
                );
                if (!listing) {
                    throw new BankError(`Seller has not listed asset ${assetId} for sale`, 'SALE_LISTING_REQUIRED', 409);
                }
                if (listing.currency !== normalizedCurrency) {
                    throw new BankError('Quote currency does not match seller listing', 'CURRENCY_MISMATCH', 409);
                }
                const resourceLine = normalizedResourceLines?.find((line) => line.assetId === assetId);
                if (resourceLine) {
                    if (asset.kind !== 'resource_lot') {
                        throw new BankError(
                            `Asset ${assetId} is not a divisible resource lot`,
                            'NOT_A_RESOURCE_LOT'
                        );
                    }
                    if (resourceLine.quantity > asset.quantity + 1e-9) {
                        throw new BankError(
                            `Requested quantity exceeds resource lot ${assetId}`,
                            'QUANTITY_EXCEEDS_LOT',
                            409
                        );
                    }
                    resourceLine.unit = asset.unit;
                    minimumGrossTotal += listing.minimum_gross_amount * (resourceLine.quantity / asset.quantity);
                } else {
                    minimumGrossTotal += listing.minimum_gross_amount;
                }
                listingIds.push(listing.id);
            }
            if (grossAmount + 1e-9 < minimumGrossTotal) {
                throw new BankError('Quote is below the seller-authorized minimum', 'QUOTE_BELOW_MINIMUM', 409);
            }
            try {
                await this._run(
                    `INSERT INTO bank_warehouse_quote
                     (id, client_quote_id, warehouse_account_id, seller_account_id, asset_ids_json,
                      listing_ids_json, gross_amount, commission_amount, currency, expires_at, terms_json)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                    [
                        quoteId, clientId, warehouseId, sellerId, JSON.stringify(normalizedAssetIds),
                        JSON.stringify(listingIds), grossAmount, commissionAmount, normalizedCurrency,
                        expiresAt,
                        JSON.stringify({
                            ...terms,
                            ...(normalizedResourceLines ? { resourceLines: normalizedResourceLines } : {})
                        })
                    ]
                );
            } catch (error) {
                if (String(error.message).includes('UNIQUE constraint failed')) {
                    const existing = await this._get(
                        `SELECT * FROM bank_warehouse_quote
                         WHERE warehouse_account_id = ? AND client_quote_id = ?`,
                        [warehouseId, clientId]
                    );
                    return mapWarehouseQuote(existing);
                }
                throw error;
            }
            return this.getWarehouseQuote(quoteId);
        });
    }

    async getWarehouseQuote(quoteId) {
        const id = requireUuid(quoteId, 'quoteId');
        const row = await this._get('SELECT * FROM bank_warehouse_quote WHERE id = ?', [id]);
        if (!row) throw new BankError('Warehouse quote not found', 'QUOTE_NOT_FOUND', 404);
        return mapWarehouseQuote(row);
    }

    async createWarehouseListing({
        assetId,
        sellerAccountId,
        warehouseAccountId,
        minimumGrossAmount = 0,
        currency = 'GLC',
        expiresInSeconds = 86400,
        idempotencyKey
    }) {
        const id = requireUuid(assetId, 'assetId');
        const sellerId = await this._requireAccount(sellerAccountId, 'sellerAccountId');
        const warehouseId = await this._requireAccount(warehouseAccountId, 'warehouseAccountId');
        const key = requireString(idempotencyKey, 'idempotencyKey');
        const normalizedCurrency = requireString(currency, 'currency').toUpperCase();
        if (!Number.isFinite(minimumGrossAmount) || minimumGrossAmount < 0) {
            throw new BankError('minimumGrossAmount cannot be negative', 'INVALID_AMOUNT');
        }
        if (!Number.isInteger(expiresInSeconds) || expiresInSeconds < 60 || expiresInSeconds > 2592000) {
            throw new BankError('expiresInSeconds must be an integer from 60 to 2592000', 'INVALID_LISTING_EXPIRY');
        }
        const expiresAt = new Date(Date.now() + expiresInSeconds * 1000).toISOString();
        return this._write(async () => {
            const existing = await this._get('SELECT * FROM bank_warehouse_listing WHERE idempotency_key = ?', [key]);
            if (existing) return mapWarehouseListing(existing);
            const warehouse = await this._get('SELECT account_type FROM bank_account WHERE id = ?', [warehouseId]);
            if (warehouse.account_type !== 'warehouse') throw new BankError('Not a warehouse account', 'NOT_A_WAREHOUSE', 409);
            const asset = await this._get('SELECT * FROM bank_asset WHERE id = ?', [id]);
            if (!asset) throw new BankError('Asset not found', 'ASSET_NOT_FOUND', 404);
            if (asset.owner_account_id !== sellerId) throw new BankError('Seller does not own asset', 'OWNER_MISMATCH', 409);
            if (asset.custodian_account_id !== warehouseId || asset.status !== 'stored') {
                throw new BankError('Asset must be stored at the selected warehouse', 'WAREHOUSE_CUSTODY_REQUIRED', 409);
            }
            await this._run(
                `UPDATE bank_warehouse_listing SET status = 'cancelled'
                 WHERE asset_id = ? AND status = 'open'`,
                [id]
            );
            const listingId = uuidv4();
            await this._run(
                `INSERT INTO bank_warehouse_listing
                 (id, idempotency_key, asset_id, seller_account_id, warehouse_account_id,
                  minimum_gross_amount, currency, expires_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
                [listingId, key, id, sellerId, warehouseId, minimumGrossAmount, normalizedCurrency, expiresAt]
            );
            return mapWarehouseListing(await this._get('SELECT * FROM bank_warehouse_listing WHERE id = ?', [listingId]));
        });
    }

    async settleWarehouseQuote({ quoteId, buyerAccountId, actorAccountId, idempotencyKey }) {
        const quote = await this.getWarehouseQuote(quoteId);
        return this.settleWarehouseTrade({
            quoteId: quote.id,
            assetIds: quote.assetIds,
            sellerAccountId: quote.sellerAccountId,
            buyerAccountId,
            warehouseAccountId: quote.warehouseAccountId,
            grossAmount: quote.grossAmount,
            commissionAmount: quote.commissionAmount,
            currency: quote.currency,
            actorAccountId,
            idempotencyKey,
            metadata: { quoteTerms: quote.terms }
        });
    }

    async createAccount({ externalUserId = null, displayName, accountType = 'player', id = uuidv4() }) {
        const accountId = requireUuid(id, 'id');
        const name = requireString(displayName, 'displayName');
        if (!ACCOUNT_TYPES.has(accountType)) {
            throw new BankError(`Unsupported accountType: ${accountType}`, 'INVALID_ACCOUNT_TYPE');
        }
        if (externalUserId != null) requireString(String(externalUserId), 'externalUserId');

        return this._write(async () => {
            try {
                await this._run(
                    `INSERT INTO bank_account (id, external_user_id, display_name, account_type)
                     VALUES (?, ?, ?, ?)`,
                    [accountId, externalUserId == null ? null : String(externalUserId), name, accountType]
                );
            } catch (error) {
                if (String(error.message).includes('UNIQUE constraint failed')) {
                    throw new BankError('Account already exists', 'ACCOUNT_EXISTS', 409);
                }
                throw error;
            }
            return this.getAccount(accountId);
        });
    }

    async getAccount(accountId) {
        const id = requireUuid(accountId, 'accountId');
        const row = await this._get('SELECT * FROM bank_account WHERE id = ?', [id]);
        if (!row) throw new BankError('Account not found', 'ACCOUNT_NOT_FOUND', 404);
        return {
            id: row.id,
            externalUserId: row.external_user_id,
            displayName: row.display_name,
            accountType: row.account_type,
            createdAt: row.created_at
        };
    }

    async getAccountByExternalUserId(externalUserId) {
        const externalId = requireString(String(externalUserId), 'externalUserId');
        const row = await this._get('SELECT * FROM bank_account WHERE external_user_id = ?', [externalId]);
        if (!row) throw new BankError('Game-bank account not found', 'ACCOUNT_NOT_FOUND', 404);
        return {
            id: row.id,
            externalUserId: row.external_user_id,
            displayName: row.display_name,
            accountType: row.account_type,
            createdAt: row.created_at
        };
    }

    async resolveAccount({ externalUserId, displayName, accountType = 'player' }) {
        const externalId = requireString(String(externalUserId), 'externalUserId');
        try {
            return await this.getAccountByExternalUserId(externalId);
        } catch (error) {
            if (!(error instanceof BankError) || error.code !== 'ACCOUNT_NOT_FOUND') throw error;
        }
        return this.createAccount({ externalUserId: externalId, displayName, accountType });
    }

    async _requireAccount(accountId, name = 'accountId') {
        const id = requireUuid(accountId, name);
        const row = await this._get('SELECT id FROM bank_account WHERE id = ?', [id]);
        if (!row) throw new BankError(`${name} does not exist`, 'ACCOUNT_NOT_FOUND', 404);
        return id;
    }

    async _recordEvent({ assetId, transactionId, eventType, actorAccountId = null, details = {} }) {
        const eventId = uuidv4();
        await this._run(
            `INSERT INTO bank_asset_event
             (id, asset_id, transaction_id, event_type, actor_account_id, details_json)
             VALUES (?, ?, ?, ?, ?, ?)`,
            [eventId, assetId, transactionId, eventType, actorAccountId, JSON.stringify(details)]
        );
        return eventId;
    }

    async _recordTransaction({
        transactionId = uuidv4(),
        transactionType,
        actorAccountId = null,
        idempotencyKey = null,
        details = {}
    }) {
        await this._run(
            `INSERT INTO bank_transaction
             (id, transaction_type, idempotency_key, actor_account_id, status, details_json)
             VALUES (?, ?, ?, ?, 'posted', ?)`,
            [transactionId, transactionType, idempotencyKey, actorAccountId, JSON.stringify(details)]
        );
        return transactionId;
    }

    async registerParcel({
        tileId,
        ownerAccountId,
        actorAccountId = ownerAccountId,
        metadata = {},
        saleStatus = 'held',
        priceAmount = null,
        priceCurrency = 'GLC'
    }) {
        const tile = parseTileId(tileId);
        const ownerId = await this._requireAccount(ownerAccountId, 'ownerAccountId');
        const actorId = actorAccountId == null ? null : await this._requireAccount(actorAccountId, 'actorAccountId');
        if (!['held', 'for_sale'].includes(saleStatus)) {
            throw new BankError('saleStatus must be held or for_sale', 'INVALID_SALE_STATUS');
        }
        if (saleStatus === 'for_sale' && (!Number.isFinite(priceAmount) || priceAmount <= 0)) {
            throw new BankError('A parcel for sale requires a positive priceAmount', 'PARCEL_PRICE_REQUIRED');
        }
        if (priceAmount != null && (!Number.isFinite(priceAmount) || priceAmount <= 0)) {
            throw new BankError('priceAmount must be positive', 'INVALID_AMOUNT');
        }
        const normalizedCurrency = requireString(priceCurrency, 'priceCurrency').toUpperCase();

        return this._write(async () => {
            const existing = await this._get('SELECT tile_id FROM bank_parcel WHERE tile_id = ?', [tile.tileId]);
            if (existing) throw new BankError('Parcel is already registered', 'PARCEL_EXISTS', 409);

            const titleAssetId = uuidv4();
            const transactionId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType: 'parcel_registration',
                actorAccountId: actorId,
                details: { tileId: tile.tileId, ownerAccountId: ownerId, saleStatus, priceAmount, priceCurrency: normalizedCurrency }
            });
            await this._run(
                `INSERT INTO bank_asset
                 (id, kind, asset_type, owner_account_id, status, origin_tile_id, metadata_json)
                 VALUES (?, 'land_title', 'terrain_parcel', ?, 'active', ?, ?)`,
                [titleAssetId, ownerId, tile.tileId, JSON.stringify(metadata)]
            );
            await this._run(
                `INSERT INTO bank_parcel
                 (tile_id, depth, col, row, title_asset_id, sale_status, price_amount, price_currency, metadata_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                    tile.tileId, tile.depth, tile.col, tile.row, titleAssetId, saleStatus,
                    priceAmount, normalizedCurrency, JSON.stringify(metadata)
                ]
            );
            await this._recordEvent({
                assetId: titleAssetId,
                transactionId,
                eventType: 'parcel_registered',
                actorAccountId: actorId,
                details: { tileId: tile.tileId, ownerAccountId: ownerId, saleStatus, priceAmount, priceCurrency: normalizedCurrency }
            });
            return this.getParcel(tile.tileId);
        });
    }

    async getParcel(tileId) {
        const tile = parseTileId(tileId);
        const row = await this._get(
            `SELECT p.*, a.owner_account_id, o.display_name AS owner_name
             FROM bank_parcel p
             JOIN bank_asset a ON a.id = p.title_asset_id
             JOIN bank_account o ON o.id = a.owner_account_id
             WHERE p.tile_id = ?`,
            [tile.tileId]
        );
        if (!row) throw new BankError('Parcel not found', 'PARCEL_NOT_FOUND', 404);
        return {
            tileId: row.tile_id,
            depth: row.depth,
            col: row.col,
            row: row.row,
            titleAssetId: row.title_asset_id,
            ownerAccountId: row.owner_account_id,
            ownerName: row.owner_name,
            saleStatus: row.sale_status,
            priceAmount: row.price_amount,
            priceCurrency: row.price_currency,
            metadata: parseJson(row.metadata_json),
            createdAt: row.created_at
        };
    }

    async getParcels(tileIds) {
        if (!Array.isArray(tileIds) || tileIds.length === 0) return [];
        const normalized = tileIds.map((tileId) => parseTileId(tileId).tileId);
        const placeholders = normalized.map(() => '?').join(',');
        const rows = await this._all(
            `SELECT p.*, a.owner_account_id, o.display_name AS owner_name
             FROM bank_parcel p
             JOIN bank_asset a ON a.id = p.title_asset_id
             JOIN bank_account o ON o.id = a.owner_account_id
             WHERE p.tile_id IN (${placeholders})`,
            normalized
        );
        return rows.map((row) => ({
            tileId: row.tile_id,
            titleAssetId: row.title_asset_id,
            ownerAccountId: row.owner_account_id,
            ownerName: row.owner_name,
            saleStatus: row.sale_status,
            priceAmount: row.price_amount,
            priceCurrency: row.price_currency,
            metadata: parseJson(row.metadata_json)
        }));
    }

    async purchaseParcels({
        tileIds,
        buyerAccountId,
        actorAccountId = buyerAccountId,
        idempotencyKey
    }) {
        if (!Array.isArray(tileIds) || tileIds.length === 0 || tileIds.length > 100) {
            throw new BankError('tileIds must contain between 1 and 100 terrain tile IDs', 'INVALID_PARCEL_LIST');
        }
        const normalizedTileIds = [...new Set(tileIds.map((tileId) => parseTileId(tileId).tileId))];
        if (normalizedTileIds.length !== tileIds.length) {
            throw new BankError('tileIds cannot contain duplicates', 'DUPLICATE_PARCEL');
        }
        const buyerId = await this._requireAccount(buyerAccountId, 'buyerAccountId');
        const actorId = await this._requireAccount(actorAccountId, 'actorAccountId');
        const key = requireString(idempotencyKey, 'idempotencyKey');

        return this._write(async () => {
            const existing = await this._findTransactionByIdempotencyKey(key);
            if (existing) {
                if (existing.transactionType !== 'parcel_purchase') {
                    throw new BankError('Idempotency key belongs to another operation', 'IDEMPOTENCY_CONFLICT', 409);
                }
                return {
                    transaction: existing,
                    parcels: await Promise.all(normalizedTileIds.map((tileId) => this.getParcel(tileId)))
                };
            }

            const parcels = [];
            let currency = null;
            let total = 0;
            const sellerCredits = new Map();
            for (const tileId of normalizedTileIds) {
                const parcel = await this._get(
                    `SELECT p.*, a.owner_account_id, a.status AS asset_status, a.version AS asset_version
                     FROM bank_parcel p JOIN bank_asset a ON a.id = p.title_asset_id
                     WHERE p.tile_id = ?`,
                    [tileId]
                );
                if (!parcel) throw new BankError(`Parcel not found: ${tileId}`, 'PARCEL_NOT_FOUND', 404);
                if (parcel.sale_status !== 'for_sale' || !Number.isFinite(parcel.price_amount) || parcel.price_amount <= 0) {
                    throw new BankError(`Parcel is not for sale: ${tileId}`, 'PARCEL_NOT_FOR_SALE', 409);
                }
                if (parcel.owner_account_id === buyerId) {
                    throw new BankError(`Buyer already owns parcel: ${tileId}`, 'NO_OWNERSHIP_CHANGE', 409);
                }
                if (parcel.asset_status !== 'active') {
                    throw new BankError(`Parcel title is not active: ${tileId}`, 'ASSET_NOT_TRANSFERABLE', 409);
                }
                currency ??= parcel.price_currency;
                if (parcel.price_currency !== currency) {
                    throw new BankError('One purchase cannot combine parcel currencies', 'CURRENCY_MISMATCH', 409);
                }
                total += parcel.price_amount;
                sellerCredits.set(
                    parcel.owner_account_id,
                    (sellerCredits.get(parcel.owner_account_id) || 0) + parcel.price_amount
                );
                parcels.push(parcel);
            }
            await this._assertDebitAllowed(buyerId, currency, -total);

            const transactionId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType: 'parcel_purchase',
                actorAccountId: actorId,
                idempotencyKey: key,
                details: { tileIds: normalizedTileIds, buyerAccountId: buyerId, total, currency }
            });
            await this._run(
                `INSERT INTO bank_ledger_entry
                 (id, transaction_id, account_id, currency, amount, entry_role)
                 VALUES (?, ?, ?, ?, ?, 'parcel_buyer_payment')`,
                [uuidv4(), transactionId, buyerId, currency, -total]
            );
            for (const [sellerId, amount] of sellerCredits) {
                await this._run(
                    `INSERT INTO bank_ledger_entry
                     (id, transaction_id, account_id, currency, amount, entry_role)
                     VALUES (?, ?, ?, ?, ?, 'parcel_seller_proceeds')`,
                    [uuidv4(), transactionId, sellerId, currency, amount]
                );
            }
            for (const parcel of parcels) {
                const update = await this._run(
                    `UPDATE bank_asset SET owner_account_id = ?, version = version + 1, updated_at = CURRENT_TIMESTAMP
                     WHERE id = ? AND owner_account_id = ? AND version = ?`,
                    [buyerId, parcel.title_asset_id, parcel.owner_account_id, parcel.asset_version]
                );
                if (update.changes !== 1) {
                    throw new BankError('Parcel ownership changed concurrently', 'OWNERSHIP_CONFLICT', 409);
                }
                await this._run(
                    `UPDATE bank_parcel SET sale_status = 'owned' WHERE tile_id = ?`,
                    [parcel.tile_id]
                );
                await this._recordEvent({
                    assetId: parcel.title_asset_id,
                    transactionId,
                    eventType: 'parcel_purchased',
                    actorAccountId: actorId,
                    details: {
                        tileId: parcel.tile_id,
                        fromOwnerAccountId: parcel.owner_account_id,
                        toOwnerAccountId: buyerId,
                        priceAmount: parcel.price_amount,
                        priceCurrency: currency
                    }
                });
            }
            return {
                transaction: await this.getTransaction(transactionId),
                parcels: await Promise.all(normalizedTileIds.map((tileId) => this.getParcel(tileId)))
            };
        });
    }

    async issueAsset({
        kind,
        assetType,
        ownerAccountId,
        custodianAccountId = null,
        quantity = null,
        unit = null,
        originTileId = null,
        locationTileId = null,
        metadata = {},
        actorAccountId = ownerAccountId,
        id = uuidv4(),
        sourceNamespace = null,
        sourceId = null
    }) {
        if (!ASSET_KINDS.has(kind) || kind === 'land_title') {
            throw new BankError(`Unsupported issuable asset kind: ${kind}`, 'INVALID_ASSET_KIND');
        }
        const assetId = requireUuid(id, 'id');
        const normalizedType = requireString(assetType, 'assetType');
        const source = sourceNamespace == null && sourceId == null ? null : {
            namespace: requireString(sourceNamespace, 'sourceNamespace'),
            id: requireString(sourceId, 'sourceId')
        };
        if (source && kind === 'resource_lot') {
            throw new BankError('Source bindings identify serialized instances, not repeatable production', 'INVALID_SOURCE_BINDING');
        }
        const ownerId = await this._requireAccount(ownerAccountId, 'ownerAccountId');
        const custodianId = custodianAccountId == null
            ? null
            : await this._requireAccount(custodianAccountId, 'custodianAccountId');
        const actorId = actorAccountId == null ? null : await this._requireAccount(actorAccountId, 'actorAccountId');
        const normalizedOrigin = originTileId == null ? null : parseTileId(originTileId).tileId;
        const normalizedLocation = locationTileId == null ? null : parseTileId(locationTileId).tileId;

        if (kind === 'resource_lot') {
            if (!Number.isFinite(quantity) || quantity <= 0) {
                throw new BankError('Resource lots require a positive quantity', 'INVALID_QUANTITY');
            }
            requireString(unit, 'unit');
            if (!normalizedOrigin) {
                throw new BankError('Resource lots require originTileId', 'ORIGIN_REQUIRED');
            }
        } else if (quantity != null) {
            throw new BankError('Serialized assets cannot have quantity', 'SERIAL_ASSET_QUANTITY');
        }

        return this._write(async () => {
            for (const [tileId, code] of [[normalizedOrigin, 'ORIGIN_NOT_REGISTERED'], [normalizedLocation, 'LOCATION_NOT_REGISTERED']]) {
                if (!tileId) continue;
                const parcel = await this._get('SELECT tile_id FROM bank_parcel WHERE tile_id = ?', [tileId]);
                if (!parcel) throw new BankError(`Terrain parcel is not registered: ${tileId}`, code, 409);
            }
            if (source) {
                const binding = await this._get('SELECT asset_id FROM bank_source_binding WHERE namespace=? AND source_id=?',
                    [source.namespace, source.id]);
                if (binding) {
                    const existing = await this.getAsset(binding.asset_id);
                    if (existing.kind !== kind || existing.assetType !== normalizedType
                        || existing.ownerAccountId !== ownerId) {
                        throw new BankError('Source instance already has different registration/ownership', 'SOURCE_BINDING_CONFLICT', 409);
                    }
                    return existing;
                }
            }
            const transactionId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType: kind === 'resource_lot' ? 'resource_issuance' : 'asset_issuance',
                actorAccountId: actorId,
                details: {
                    kind, assetType: normalizedType, ownerAccountId: ownerId, quantity, unit,
                    originTileId: normalizedOrigin, locationTileId: normalizedLocation
                }
            });
            await this._run(
                `INSERT INTO bank_asset
                 (id, kind, asset_type, owner_account_id, custodian_account_id, quantity, unit,
                  status, origin_tile_id, location_tile_id, metadata_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                    assetId, kind, normalizedType, ownerId, custodianId, quantity, unit,
                    normalizedLocation && kind !== 'resource_lot' ? 'deployed' : 'active',
                    normalizedOrigin, normalizedLocation, JSON.stringify(metadata)
                ]
            );
            await this._recordEvent({
                assetId,
                transactionId,
                eventType: kind === 'resource_lot' ? 'resource_issued' : 'asset_issued',
                actorAccountId: actorId,
                details: {
                    ownerAccountId: ownerId,
                    custodianAccountId: custodianId,
                    quantity,
                    unit,
                    originTileId: normalizedOrigin,
                    locationTileId: normalizedLocation,
                    metadata
                }
            });
            if (source) {
                await this._run('INSERT INTO bank_source_binding(namespace,source_id,asset_id) VALUES (?,?,?)',
                    [source.namespace, source.id, assetId]);
            }
            return this.getAsset(assetId);
        });
    }

    async getAsset(assetId) {
        const id = requireUuid(assetId, 'assetId');
        const row = await this._get(
            `SELECT a.*, o.display_name AS owner_name, c.display_name AS custodian_name
             FROM bank_asset a
             JOIN bank_account o ON o.id = a.owner_account_id
             LEFT JOIN bank_account c ON c.id = a.custodian_account_id
             WHERE a.id = ?`,
            [id]
        );
        if (!row) throw new BankError('Asset not found', 'ASSET_NOT_FOUND', 404);
        return mapAsset(row);
    }

    async verifyAsset(assetId) {
        try {
            const asset = await this.getAsset(assetId);
            return {
                authentic: true,
                spendable: ACTIVE_ASSET_STATES.has(asset.status),
                asset,
                reason: ACTIVE_ASSET_STATES.has(asset.status)
                    ? 'Asset exists in the authoritative registry and is active'
                    : `Asset is authentic but its status is ${asset.status}`
            };
        } catch (error) {
            if (error instanceof BankError && error.code === 'ASSET_NOT_FOUND') {
                return { authentic: false, spendable: false, asset: null, reason: 'UUID is not registered' };
            }
            throw error;
        }
    }

    async transferAsset({ assetId, fromOwnerAccountId, toOwnerAccountId, actorAccountId = fromOwnerAccountId }) {
        const id = requireUuid(assetId, 'assetId');
        const fromId = await this._requireAccount(fromOwnerAccountId, 'fromOwnerAccountId');
        const toId = await this._requireAccount(toOwnerAccountId, 'toOwnerAccountId');
        const actorId = actorAccountId == null ? null : await this._requireAccount(actorAccountId, 'actorAccountId');
        if (fromId === toId) throw new BankError('Owner is unchanged', 'NO_OWNERSHIP_CHANGE');

        return this._write(async () => {
            const asset = await this._get('SELECT * FROM bank_asset WHERE id = ?', [id]);
            if (!asset) throw new BankError('Asset not found', 'ASSET_NOT_FOUND', 404);
            if (asset.owner_account_id !== fromId) {
                throw new BankError('fromOwnerAccountId is not the current owner', 'OWNER_MISMATCH', 409);
            }
            if (!ACTIVE_ASSET_STATES.has(asset.status)) {
                throw new BankError(`Asset cannot be transferred while ${asset.status}`, 'ASSET_NOT_TRANSFERABLE', 409);
            }
            const transactionId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType: 'asset_transfer',
                actorAccountId: actorId,
                details: { assetId: id, fromOwnerAccountId: fromId, toOwnerAccountId: toId }
            });
            await this._run(
                `UPDATE bank_asset
                 SET owner_account_id = ?, version = version + 1, updated_at = CURRENT_TIMESTAMP
                 WHERE id = ? AND version = ?`,
                [toId, id, asset.version]
            );
            await this._recordEvent({
                assetId: id,
                transactionId,
                eventType: 'ownership_transferred',
                actorAccountId: actorId,
                details: { fromOwnerAccountId: fromId, toOwnerAccountId: toId }
            });
            return this.getAsset(id);
        });
    }

    async changeCustody({ assetId, ownerAccountId, toCustodianAccountId = null, actorAccountId = ownerAccountId }) {
        const id = requireUuid(assetId, 'assetId');
        const ownerId = await this._requireAccount(ownerAccountId, 'ownerAccountId');
        const custodianId = toCustodianAccountId == null
            ? null
            : await this._requireAccount(toCustodianAccountId, 'toCustodianAccountId');
        const actorId = actorAccountId == null ? null : await this._requireAccount(actorAccountId, 'actorAccountId');
        if (custodianId) {
            const custodian = await this._get('SELECT account_type FROM bank_account WHERE id = ?', [custodianId]);
            if (custodian.account_type !== 'warehouse') {
                throw new BankError('Custody deposits require a warehouse account', 'NOT_A_WAREHOUSE', 409);
            }
        }

        return this._write(async () => {
            const asset = await this._get('SELECT * FROM bank_asset WHERE id = ?', [id]);
            if (!asset) throw new BankError('Asset not found', 'ASSET_NOT_FOUND', 404);
            if (asset.owner_account_id !== ownerId) {
                throw new BankError('Only the current owner can authorize custody', 'OWNER_MISMATCH', 409);
            }
            if (!ACTIVE_ASSET_STATES.has(asset.status)) {
                throw new BankError(`Asset cannot move while ${asset.status}`, 'ASSET_NOT_MOVABLE', 409);
            }
            const fromCustodianAccountId = asset.custodian_account_id;
            if (fromCustodianAccountId === custodianId) {
                throw new BankError('Custodian is unchanged', 'NO_CUSTODY_CHANGE');
            }
            const transactionId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType: 'custody_change',
                actorAccountId: actorId,
                details: { assetId: id, fromCustodianAccountId, toCustodianAccountId: custodianId }
            });
            await this._run(
                `UPDATE bank_asset
                 SET custodian_account_id = ?, status = ?, version = version + 1,
                     updated_at = CURRENT_TIMESTAMP
                 WHERE id = ?`,
                [custodianId, custodianId ? 'stored' : 'active', id]
            );
            await this._recordEvent({
                assetId: id,
                transactionId,
                eventType: custodianId ? 'custody_deposited' : 'custody_released',
                actorAccountId: actorId,
                details: { fromCustodianAccountId, toCustodianAccountId: custodianId, ownerAccountId: ownerId }
            });
            return this.getAsset(id);
        });
    }

    async splitResourceLot({ assetId, ownerAccountId, quantities, actorAccountId = ownerAccountId }) {
        const id = requireUuid(assetId, 'assetId');
        const ownerId = await this._requireAccount(ownerAccountId, 'ownerAccountId');
        const actorId = actorAccountId == null ? null : await this._requireAccount(actorAccountId, 'actorAccountId');
        if (!Array.isArray(quantities) || quantities.length < 2 || quantities.length > 100) {
            throw new BankError('quantities must contain between 2 and 100 child quantities', 'INVALID_SPLIT');
        }
        if (quantities.some((quantity) => !Number.isFinite(quantity) || quantity <= 0)) {
            throw new BankError('Every child quantity must be positive', 'INVALID_QUANTITY');
        }

        return this._write(async () => {
            const parent = await this._get('SELECT * FROM bank_asset WHERE id = ?', [id]);
            if (!parent) throw new BankError('Asset not found', 'ASSET_NOT_FOUND', 404);
            if (parent.kind !== 'resource_lot') {
                throw new BankError('Only resource lots can be split', 'NOT_A_RESOURCE_LOT');
            }
            if (parent.owner_account_id !== ownerId) {
                throw new BankError('Only the current owner can split this lot', 'OWNER_MISMATCH', 409);
            }
            if (!ACTIVE_ASSET_STATES.has(parent.status)) {
                throw new BankError(`Resource lot cannot be split while ${parent.status}`, 'ASSET_NOT_SPLITTABLE', 409);
            }
            const childTotal = quantities.reduce((sum, quantity) => sum + quantity, 0);
            const tolerance = Math.max(1e-9, Math.abs(parent.quantity) * 1e-9);
            if (Math.abs(childTotal - parent.quantity) > tolerance) {
                throw new BankError(
                    `Child quantities must conserve the parent quantity (${parent.quantity} ${parent.unit})`,
                    'QUANTITY_NOT_CONSERVED',
                    409
                );
            }

            const transactionId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType: 'resource_split',
                actorAccountId: actorId,
                details: { parentAssetId: id, quantities, unit: parent.unit }
            });
            await this._run(
                `UPDATE bank_asset
                 SET status = 'consumed', version = version + 1, updated_at = CURRENT_TIMESTAMP
                 WHERE id = ?`,
                [id]
            );
            await this._recordEvent({
                assetId: id,
                transactionId,
                eventType: 'resource_split',
                actorAccountId: actorId,
                details: { quantities, unit: parent.unit }
            });

            const childIds = [];
            for (const quantity of quantities) {
                const childId = uuidv4();
                childIds.push(childId);
                await this._run(
                    `INSERT INTO bank_asset
                     (id, kind, asset_type, owner_account_id, custodian_account_id, quantity, unit,
                      status, origin_tile_id, location_tile_id, metadata_json)
                     VALUES (?, 'resource_lot', ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                    [
                        childId,
                        parent.asset_type,
                        parent.owner_account_id,
                        parent.custodian_account_id,
                        quantity,
                        parent.unit,
                        parent.status === 'stored' ? 'stored' : 'active',
                        parent.origin_tile_id,
                        parent.location_tile_id,
                        parent.metadata_json
                    ]
                );
                await this._run(
                    `INSERT INTO bank_asset_lineage (parent_asset_id, child_asset_id, quantity)
                     VALUES (?, ?, ?)`,
                    [id, childId, quantity]
                );
                await this._recordEvent({
                    assetId: childId,
                    transactionId,
                    eventType: 'resource_created_from_split',
                    actorAccountId: actorId,
                    details: { parentAssetId: id, quantity, unit: parent.unit }
                });
            }
            return {
                transactionId,
                parent: await this.getAsset(id),
                children: await Promise.all(childIds.map((childId) => this.getAsset(childId)))
            };
        });
    }

    async getBalance(accountId, currency = 'GLC') {
        const account = await this.getAccount(accountId);
        const normalizedCurrency = requireString(currency, 'currency').toUpperCase();
        const row = await this._get(
            `SELECT COALESCE(SUM(amount), 0) AS balance
             FROM bank_ledger_entry
             WHERE account_id = ? AND currency = ?`,
            [account.id, normalizedCurrency]
        );
        return {
            accountId: account.id,
            displayName: account.displayName,
            currency: normalizedCurrency,
            balance: Number(row?.balance || 0)
        };
    }

    async getTransaction(transactionId) {
        const id = requireUuid(transactionId, 'transactionId');
        const row = await this._get('SELECT * FROM bank_transaction WHERE id = ?', [id]);
        if (!row) throw new BankError('Transaction not found', 'TRANSACTION_NOT_FOUND', 404);
        const entries = await this._all(
            `SELECT id, account_id, currency, amount, entry_role, created_at
             FROM bank_ledger_entry WHERE transaction_id = ? ORDER BY id`,
            [id]
        );
        return {
            id: row.id,
            transactionType: row.transaction_type,
            idempotencyKey: row.idempotency_key,
            actorAccountId: row.actor_account_id,
            status: row.status,
            details: parseJson(row.details_json),
            entries: entries.map((entry) => ({
                id: entry.id,
                accountId: entry.account_id,
                currency: entry.currency,
                amount: entry.amount,
                entryRole: entry.entry_role,
                createdAt: entry.created_at
            })),
            createdAt: row.created_at
        };
    }

    async _findTransactionByIdempotencyKey(idempotencyKey) {
        const row = await this._get(
            'SELECT id FROM bank_transaction WHERE idempotency_key = ?',
            [idempotencyKey]
        );
        return row ? this.getTransaction(row.id) : null;
    }

    async _assertDebitAllowed(accountId, currency, debitAmount) {
        const account = await this._get('SELECT * FROM bank_account WHERE id = ?', [accountId]);
        if (!account) throw new BankError('Debit account not found', 'ACCOUNT_NOT_FOUND', 404);
        if (account.account_type === 'treasury' || account.account_type === 'system') return;
        const balance = await this._get(
            `SELECT COALESCE(SUM(amount), 0) AS balance
             FROM bank_ledger_entry WHERE account_id = ? AND currency = ?`,
            [accountId, currency]
        );
        if (Number(balance?.balance || 0) + debitAmount < -1e-9) {
            throw new BankError('Insufficient bank balance', 'INSUFFICIENT_FUNDS', 409);
        }
    }

    async transferCredits({
        fromAccountId,
        toAccountId,
        amount,
        currency = 'GLC',
        actorAccountId = fromAccountId,
        idempotencyKey,
        transactionType = 'credit_transfer',
        metadata = {}
    }) {
        const fromId = await this._requireAccount(fromAccountId, 'fromAccountId');
        const toId = await this._requireAccount(toAccountId, 'toAccountId');
        const actorId = actorAccountId == null ? null : await this._requireAccount(actorAccountId, 'actorAccountId');
        const normalizedCurrency = requireString(currency, 'currency').toUpperCase();
        const key = requireString(idempotencyKey, 'idempotencyKey');
        if (fromId === toId) throw new BankError('Credit accounts must be different', 'NO_BALANCE_CHANGE');
        if (!Number.isFinite(amount) || amount <= 0) {
            throw new BankError('amount must be positive', 'INVALID_AMOUNT');
        }

        return this._write(async () => {
            const existing = await this._findTransactionByIdempotencyKey(key);
            if (existing) {
                const terms = existing.details;
                if (existing.transactionType !== transactionType
                    || existing.actorAccountId !== actorId
                    || terms.fromAccountId !== fromId || terms.toAccountId !== toId
                    || terms.amount !== amount || terms.currency !== normalizedCurrency) {
                    throw new BankError('Idempotency key belongs to different credit transfer terms',
                        'IDEMPOTENCY_CONFLICT', 409);
                }
                return existing;
            }
            await this._assertDebitAllowed(fromId, normalizedCurrency, -amount);

            const transactionId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType,
                actorAccountId: actorId,
                idempotencyKey: key,
                details: { fromAccountId: fromId, toAccountId: toId, amount, currency: normalizedCurrency, metadata }
            });
            await this._run(
                `INSERT INTO bank_ledger_entry
                 (id, transaction_id, account_id, currency, amount, entry_role)
                 VALUES (?, ?, ?, ?, ?, 'debit'), (?, ?, ?, ?, ?, 'credit')`,
                [
                    uuidv4(), transactionId, fromId, normalizedCurrency, -amount,
                    uuidv4(), transactionId, toId, normalizedCurrency, amount
                ]
            );
            return this.getTransaction(transactionId);
        });
    }

    async settleWarehouseTrade({
        assetIds,
        sellerAccountId,
        buyerAccountId,
        warehouseAccountId,
        grossAmount,
        commissionAmount = 0,
        currency = 'GLC',
        actorAccountId,
        idempotencyKey,
        quoteId,
        metadata = {}
    }) {
        if (!Array.isArray(assetIds) || assetIds.length === 0 || assetIds.length > 100) {
            throw new BankError('assetIds must contain between 1 and 100 UUIDs', 'INVALID_ASSET_LIST');
        }
        const normalizedAssetIds = [...new Set(assetIds.map((id) => requireUuid(id, 'assetId')))];
        if (normalizedAssetIds.length !== assetIds.length) {
            throw new BankError('assetIds cannot contain duplicates', 'DUPLICATE_ASSET');
        }
        const sellerId = await this._requireAccount(sellerAccountId, 'sellerAccountId');
        const buyerId = await this._requireAccount(buyerAccountId, 'buyerAccountId');
        const warehouseId = await this._requireAccount(warehouseAccountId, 'warehouseAccountId');
        const actorId = await this._requireAccount(actorAccountId, 'actorAccountId');
        const normalizedCurrency = requireString(currency, 'currency').toUpperCase();
        const key = requireString(idempotencyKey, 'idempotencyKey');
        const normalizedQuoteId = requireUuid(quoteId, 'quoteId');
        if (sellerId === buyerId) throw new BankError('Buyer and seller must be different', 'INVALID_TRADE_PARTIES');
        if (!Number.isFinite(grossAmount) || grossAmount <= 0) {
            throw new BankError('grossAmount must be positive', 'INVALID_AMOUNT');
        }
        if (!Number.isFinite(commissionAmount) || commissionAmount < 0 || commissionAmount > grossAmount) {
            throw new BankError('commissionAmount must be between zero and grossAmount', 'INVALID_COMMISSION');
        }

        return this._write(async () => {
            const existing = await this._findTransactionByIdempotencyKey(key);
            if (existing) {
                if (!['warehouse_trade', 'warehouse_partial_trade'].includes(existing.transactionType)) {
                    throw new BankError('Idempotency key belongs to another operation', 'IDEMPOTENCY_CONFLICT', 409);
                }
                const existingSourceIds = existing.details.sourceAssetIds || existing.details.assetIds || [];
                const sameRequest = existing.details.quoteId === normalizedQuoteId
                    && existingSourceIds.length === normalizedAssetIds.length
                    && existingSourceIds.every((assetId) => normalizedAssetIds.includes(assetId));
                if (!sameRequest) {
                    throw new BankError(
                        'Idempotency key belongs to a different warehouse settlement',
                        'IDEMPOTENCY_CONFLICT',
                        409
                    );
                }
                const settledAssetIds = existing.details.assetIds || normalizedAssetIds;
                const remainderAssetIds = existing.details.remainderAssetIds || [];
                return {
                    transaction: existing,
                    assets: await Promise.all(settledAssetIds.map((id) => this.getAsset(id))),
                    remainderAssets: await Promise.all(remainderAssetIds.map((id) => this.getAsset(id)))
                };
            }

            const quote = await this._get('SELECT * FROM bank_warehouse_quote WHERE id = ?', [normalizedQuoteId]);
            if (!quote) throw new BankError('Warehouse quote not found', 'QUOTE_NOT_FOUND', 404);
            if (quote.status !== 'open') {
                throw new BankError(`Warehouse quote is ${quote.status}`, 'QUOTE_NOT_OPEN', 409);
            }
            if (Date.parse(quote.expires_at) <= Date.now()) {
                await this._run(
                    `UPDATE bank_warehouse_quote SET status = 'expired' WHERE id = ? AND status = 'open'`,
                    [normalizedQuoteId]
                );
                throw new BankError('Warehouse quote has expired', 'QUOTE_EXPIRED', 409);
            }
            const quotedAssetIds = parseJson(quote.asset_ids_json);
            const quotedListingIds = parseJson(quote.listing_ids_json);
            const quoteTerms = parseJson(quote.terms_json);
            const resourceLines = Array.isArray(quoteTerms.resourceLines) ? quoteTerms.resourceLines : [];
            const resourceLineByAssetId = new Map(resourceLines.map((line) => [line.assetId, line]));
            const sameAssets = Array.isArray(quotedAssetIds)
                && quotedAssetIds.length === normalizedAssetIds.length
                && quotedAssetIds.every((id) => normalizedAssetIds.includes(id));
            const sameTerms = quote.seller_account_id === sellerId
                && quote.warehouse_account_id === warehouseId
                && quote.currency === normalizedCurrency
                && Math.abs(quote.gross_amount - grossAmount) < 1e-9
                && Math.abs(quote.commission_amount - commissionAmount) < 1e-9;
            if (!sameAssets || !sameTerms) {
                throw new BankError('Settlement terms do not match the registered quote', 'QUOTE_MISMATCH', 409);
            }
            if (!Array.isArray(quotedListingIds) || quotedListingIds.length !== normalizedAssetIds.length) {
                throw new BankError('Quote has no valid seller listings', 'SALE_LISTING_REQUIRED', 409);
            }
            for (let index = 0; index < quotedListingIds.length; index += 1) {
                const listing = await this._get(
                    `SELECT * FROM bank_warehouse_listing WHERE id = ?`,
                    [quotedListingIds[index]]
                );
                if (!listing || listing.asset_id !== normalizedAssetIds[index]
                    || listing.seller_account_id !== sellerId
                    || listing.warehouse_account_id !== warehouseId
                    || listing.status !== 'open'
                    || Date.parse(listing.expires_at) <= Date.now()) {
                    throw new BankError('Seller listing is no longer valid', 'SALE_LISTING_NOT_OPEN', 409);
                }
            }

            const warehouse = await this._get('SELECT account_type FROM bank_account WHERE id = ?', [warehouseId]);
            if (warehouse.account_type !== 'warehouse') {
                throw new BankError('warehouseAccountId is not a registered warehouse', 'NOT_A_WAREHOUSE', 409);
            }
            await this._assertDebitAllowed(buyerId, normalizedCurrency, -grossAmount);

            const assets = [];
            for (const assetId of normalizedAssetIds) {
                const asset = await this._get('SELECT * FROM bank_asset WHERE id = ?', [assetId]);
                if (!asset) throw new BankError(`Asset not found: ${assetId}`, 'ASSET_NOT_FOUND', 404);
                if (asset.owner_account_id !== sellerId) {
                    throw new BankError(`Seller does not own asset ${assetId}`, 'OWNER_MISMATCH', 409);
                }
                if (asset.custodian_account_id !== warehouseId || asset.status !== 'stored') {
                    throw new BankError(`Asset ${assetId} is not deposited at this warehouse`, 'WAREHOUSE_CUSTODY_REQUIRED', 409);
                }
                assets.push(asset);
            }

            const transactionId = uuidv4();
            const sellerProceeds = grossAmount - commissionAmount;
            const settlementPlan = assets.map((asset) => {
                const line = resourceLineByAssetId.get(asset.id);
                if (!line) return { source: asset, soldAssetId: asset.id, remainderAssetId: null };
                if (asset.kind !== 'resource_lot' || line.unit !== asset.unit) {
                    throw new BankError('Quoted resource line no longer matches its lot', 'QUOTE_MISMATCH', 409);
                }
                const quantity = Number(line.quantity);
                if (!Number.isFinite(quantity) || quantity <= 0 || quantity > asset.quantity + 1e-9) {
                    throw new BankError('Quoted resource quantity is no longer valid', 'QUOTE_MISMATCH', 409);
                }
                if (Math.abs(quantity - asset.quantity) <= Math.max(1e-9, asset.quantity * 1e-9)) {
                    return { source: asset, soldAssetId: asset.id, remainderAssetId: null };
                }
                return {
                    source: asset,
                    quantity,
                    soldAssetId: uuidv4(),
                    remainderAssetId: uuidv4(),
                    remainderQuantity: asset.quantity - quantity
                };
            });
            const soldAssetIds = settlementPlan.map((plan) => plan.soldAssetId);
            const remainderAssetIds = settlementPlan
                .map((plan) => plan.remainderAssetId)
                .filter(Boolean);
            const hasPartialLots = remainderAssetIds.length > 0;
            await this._recordTransaction({
                transactionId,
                transactionType: hasPartialLots ? 'warehouse_partial_trade' : 'warehouse_trade',
                actorAccountId: actorId,
                idempotencyKey: key,
                details: {
                    sourceAssetIds: normalizedAssetIds,
                    assetIds: soldAssetIds,
                    remainderAssetIds,
                    resourceLines,
                    sellerAccountId: sellerId,
                    buyerAccountId: buyerId,
                    warehouseAccountId: warehouseId,
                    grossAmount,
                    commissionAmount,
                    sellerProceeds,
                    currency: normalizedCurrency,
                    quoteId: normalizedQuoteId,
                    metadata
                }
            });

            const entries = [
                [uuidv4(), transactionId, buyerId, normalizedCurrency, -grossAmount, 'buyer_payment']
            ];
            if (sellerProceeds > 0) {
                entries.push([uuidv4(), transactionId, sellerId, normalizedCurrency, sellerProceeds, 'seller_proceeds']);
            }
            if (commissionAmount > 0) {
                entries.push([uuidv4(), transactionId, warehouseId, normalizedCurrency, commissionAmount, 'warehouse_commission']);
            }
            for (const entry of entries) {
                await this._run(
                    `INSERT INTO bank_ledger_entry
                     (id, transaction_id, account_id, currency, amount, entry_role)
                     VALUES (?, ?, ?, ?, ?, ?)`,
                    entry
                );
            }

            const ledgerTotal = entries.reduce((sum, entry) => sum + entry[4], 0);
            if (Math.abs(ledgerTotal) > 1e-9) {
                throw new BankError('Warehouse ledger entries do not balance', 'UNBALANCED_TRANSACTION', 500);
            }

            for (const plan of settlementPlan) {
                const asset = plan.source;
                if (plan.remainderAssetId) {
                    await this._run(
                        `UPDATE bank_asset
                         SET status = 'consumed', version = version + 1, updated_at = CURRENT_TIMESTAMP
                         WHERE id = ? AND owner_account_id = ? AND version = ?`,
                        [asset.id, sellerId, asset.version]
                    );
                    await this._recordEvent({
                        assetId: asset.id,
                        transactionId,
                        eventType: 'resource_partially_sold',
                        actorAccountId: actorId,
                        details: {
                            soldAssetId: plan.soldAssetId,
                            soldQuantity: plan.quantity,
                            remainderAssetId: plan.remainderAssetId,
                            remainderQuantity: plan.remainderQuantity,
                            unit: asset.unit,
                            quoteId: normalizedQuoteId
                        }
                    });
                    for (const child of [
                        { id: plan.soldAssetId, ownerId: buyerId, quantity: plan.quantity, eventType: 'resource_created_for_partial_sale' },
                        { id: plan.remainderAssetId, ownerId: sellerId, quantity: plan.remainderQuantity, eventType: 'resource_remainder_created' }
                    ]) {
                        await this._run(
                            `INSERT INTO bank_asset
                             (id, kind, asset_type, owner_account_id, custodian_account_id, quantity, unit,
                              status, origin_tile_id, location_tile_id, metadata_json)
                             VALUES (?, 'resource_lot', ?, ?, ?, ?, ?, 'stored', ?, ?, ?)`,
                            [
                                child.id, asset.asset_type, child.ownerId, warehouseId, child.quantity,
                                asset.unit, asset.origin_tile_id, asset.location_tile_id, asset.metadata_json
                            ]
                        );
                        await this._run(
                            `INSERT INTO bank_asset_lineage (parent_asset_id, child_asset_id, quantity)
                             VALUES (?, ?, ?)`,
                            [asset.id, child.id, child.quantity]
                        );
                        await this._recordEvent({
                            assetId: child.id,
                            transactionId,
                            eventType: child.eventType,
                            actorAccountId: actorId,
                            details: {
                                parentAssetId: asset.id,
                                sellerAccountId: sellerId,
                                buyerAccountId: child.ownerId === buyerId ? buyerId : null,
                                warehouseAccountId: warehouseId,
                                quantity: child.quantity,
                                unit: asset.unit,
                                quoteId: normalizedQuoteId
                            }
                        });
                    }
                    continue;
                }
                await this._run(
                    `UPDATE bank_asset
                     SET owner_account_id = ?, version = version + 1, updated_at = CURRENT_TIMESTAMP
                     WHERE id = ? AND owner_account_id = ? AND version = ?`,
                    [buyerId, asset.id, sellerId, asset.version]
                );
                await this._recordEvent({
                    assetId: asset.id,
                    transactionId,
                    eventType: 'warehouse_trade_settled',
                    actorAccountId: actorId,
                    details: {
                        sellerAccountId: sellerId,
                        buyerAccountId: buyerId,
                        warehouseAccountId: warehouseId,
                        quoteId: normalizedQuoteId
                    }
                });
            }

            const quoteUpdate = await this._run(
                `UPDATE bank_warehouse_quote
                 SET status = 'settled', transaction_id = ?
                 WHERE id = ? AND status = 'open'`,
                [transactionId, normalizedQuoteId]
            );
            if (quoteUpdate.changes !== 1) {
                throw new BankError('Warehouse quote was settled concurrently', 'QUOTE_NOT_OPEN', 409);
            }
            await this._run(
                `UPDATE bank_warehouse_listing
                 SET status = 'sold', transaction_id = ?
                 WHERE id IN (${quotedListingIds.map(() => '?').join(',')}) AND status = 'open'`,
                [transactionId, ...quotedListingIds]
            );

            for (let index = 0; index < settlementPlan.length; index += 1) {
                const plan = settlementPlan[index];
                if (!plan.remainderAssetId) continue;
                const sourceListing = await this._get(
                    'SELECT * FROM bank_warehouse_listing WHERE id = ?',
                    [quotedListingIds[index]]
                );
                const remainingMinimum = sourceListing.minimum_gross_amount
                    * (plan.remainderQuantity / plan.source.quantity);
                await this._run(
                    `INSERT INTO bank_warehouse_listing
                     (id, idempotency_key, asset_id, seller_account_id, warehouse_account_id,
                      minimum_gross_amount, currency, expires_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
                    [
                        uuidv4(), `partial-remainder:${transactionId}:${plan.remainderAssetId}`,
                        plan.remainderAssetId, sellerId, warehouseId, remainingMinimum,
                        sourceListing.currency, sourceListing.expires_at
                    ]
                );
            }

            return {
                transaction: await this.getTransaction(transactionId),
                assets: await Promise.all(soldAssetIds.map((id) => this.getAsset(id))),
                remainderAssets: await Promise.all(remainderAssetIds.map((id) => this.getAsset(id)))
            };
        });
    }

    async deployAsset({ assetId, ownerAccountId, tileId, actorAccountId = ownerAccountId }) {
        const id = requireUuid(assetId, 'assetId');
        const ownerId = await this._requireAccount(ownerAccountId, 'ownerAccountId');
        const actorId = await this._requireAccount(actorAccountId, 'actorAccountId');
        const tile = parseTileId(tileId);
        return this._write(async () => {
            const parcel = await this._get(
                `SELECT a.owner_account_id FROM bank_parcel p
                 JOIN bank_asset a ON a.id = p.title_asset_id WHERE p.tile_id = ?`,
                [tile.tileId]
            );
            if (!parcel) throw new BankError('Deployment parcel is not registered', 'LOCATION_NOT_REGISTERED', 409);
            if (parcel.owner_account_id !== ownerId) {
                throw new BankError('Asset owner must own the deployment parcel', 'PARCEL_OWNER_MISMATCH', 409);
            }
            const asset = await this._get('SELECT * FROM bank_asset WHERE id = ?', [id]);
            if (!asset) throw new BankError('Asset not found', 'ASSET_NOT_FOUND', 404);
            if (asset.owner_account_id !== ownerId) throw new BankError('Asset owner mismatch', 'OWNER_MISMATCH', 409);
            if (asset.kind === 'resource_lot' || asset.kind === 'land_title') {
                throw new BankError('This asset kind cannot be deployed', 'ASSET_NOT_DEPLOYABLE', 409);
            }
            if (asset.custodian_account_id) {
                throw new BankError('An asset in custody cannot be deployed', 'ASSET_IN_CUSTODY', 409);
            }
            const transactionId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType: 'asset_deployment',
                actorAccountId: actorId,
                details: { assetId: id, tileId: tile.tileId, ownerAccountId: ownerId }
            });
            await this._run(
                `UPDATE bank_asset SET status = 'deployed', location_tile_id = ?, version = version + 1,
                 updated_at = CURRENT_TIMESTAMP WHERE id = ?`,
                [tile.tileId, id]
            );
            await this._recordEvent({
                assetId: id,
                transactionId,
                eventType: 'asset_deployed',
                actorAccountId: actorId,
                details: { tileId: tile.tileId }
            });
            return this.getAsset(id);
        });
    }

    async registerProductionRule({
        producerAssetType,
        outputAssetType,
        outputUnit,
        quantityPerHour,
        metadata = {},
        id = uuidv4()
    }) {
        const ruleId = requireUuid(id, 'id');
        const producerType = requireString(producerAssetType, 'producerAssetType');
        const outputType = requireString(outputAssetType, 'outputAssetType');
        const unit = requireString(outputUnit, 'outputUnit');
        if (!Number.isFinite(quantityPerHour) || quantityPerHour <= 0) {
            throw new BankError('quantityPerHour must be positive', 'INVALID_QUANTITY');
        }
        return this._write(async () => {
            await this._run(
                `INSERT INTO bank_production_rule
                 (id, producer_asset_type, output_asset_type, output_unit, quantity_per_hour, metadata_json)
                 VALUES (?, ?, ?, ?, ?, ?)`,
                [ruleId, producerType, outputType, unit, quantityPerHour, JSON.stringify(metadata)]
            );
            return {
                id: ruleId,
                producerAssetType: producerType,
                outputAssetType: outputType,
                outputUnit: unit,
                quantityPerHour,
                metadata
            };
        });
    }

    async claimProduction({
        ruleId,
        producerAssetId,
        ownerAccountId,
        originTileId,
        elapsedSeconds,
        sourceEventId,
        actorAccountId = ownerAccountId,
        idempotencyKey,
        evidence = {}
    }) {
        const normalizedRuleId = requireUuid(ruleId, 'ruleId');
        const producerId = requireUuid(producerAssetId, 'producerAssetId');
        const ownerId = await this._requireAccount(ownerAccountId, 'ownerAccountId');
        const actorId = await this._requireAccount(actorAccountId, 'actorAccountId');
        const tile = parseTileId(originTileId);
        const key = requireString(idempotencyKey, 'idempotencyKey');
        const sourceId = requireString(sourceEventId, 'sourceEventId');
        if (!Number.isInteger(elapsedSeconds) || elapsedSeconds <= 0 || elapsedSeconds > 86400) {
            throw new BankError('elapsedSeconds must be an integer from 1 to 86400', 'INVALID_PRODUCTION_WINDOW');
        }

        return this._write(async () => {
            const existing = await this._get(
                `SELECT output_asset_id, transaction_id FROM bank_production_claim WHERE idempotency_key = ?`,
                [key]
            );
            if (existing) {
                return {
                    transaction: await this.getTransaction(existing.transaction_id),
                    output: await this.getAsset(existing.output_asset_id)
                };
            }
            const replay = await this._get(
                `SELECT id FROM bank_production_claim WHERE source_event_id = ?`,
                [sourceId]
            );
            if (replay) throw new BankError('Production source event was already claimed', 'SOURCE_EVENT_REPLAY', 409);
            const rule = await this._get('SELECT * FROM bank_production_rule WHERE id = ? AND is_active = 1', [normalizedRuleId]);
            if (!rule) throw new BankError('Active production rule not found', 'PRODUCTION_RULE_NOT_FOUND', 404);
            const producer = await this._get('SELECT * FROM bank_asset WHERE id = ?', [producerId]);
            if (!producer) throw new BankError('Producer asset not found', 'ASSET_NOT_FOUND', 404);
            if (producer.owner_account_id !== ownerId) throw new BankError('Producer owner mismatch', 'OWNER_MISMATCH', 409);
            if (producer.asset_type !== rule.producer_asset_type) {
                throw new BankError('Producer does not satisfy this rule', 'INVALID_PRODUCER', 409);
            }
            if (producer.status !== 'deployed' || producer.location_tile_id !== tile.tileId) {
                throw new BankError('Producer must be deployed on the origin parcel', 'PRODUCER_NOT_DEPLOYED', 409);
            }
            const parcel = await this._get(
                `SELECT a.owner_account_id FROM bank_parcel p JOIN bank_asset a ON a.id = p.title_asset_id
                 WHERE p.tile_id = ?`,
                [tile.tileId]
            );
            if (!parcel || parcel.owner_account_id !== ownerId) {
                throw new BankError('Production owner must own the origin parcel', 'PARCEL_OWNER_MISMATCH', 409);
            }
            const quantity = rule.quantity_per_hour * elapsedSeconds / 3600;
            const transactionId = uuidv4();
            const outputId = uuidv4();
            await this._recordTransaction({
                transactionId,
                transactionType: 'validated_production',
                actorAccountId: actorId,
                idempotencyKey: key,
                details: {
                    ruleId: normalizedRuleId,
                    producerAssetId: producerId,
                    ownerAccountId: ownerId,
                    originTileId: tile.tileId,
                    elapsedSeconds,
                    sourceEventId: sourceId,
                    quantity,
                    unit: rule.output_unit,
                    evidence
                }
            });
            await this._run(
                `INSERT INTO bank_asset
                 (id, kind, asset_type, owner_account_id, quantity, unit, status, origin_tile_id,
                  location_tile_id, metadata_json)
                 VALUES (?, 'resource_lot', ?, ?, ?, ?, 'active', ?, ?, ?)`,
                [
                    outputId, rule.output_asset_type, ownerId, quantity, rule.output_unit,
                    tile.tileId, tile.tileId,
                    JSON.stringify({ producerAssetId: producerId, productionRuleId: normalizedRuleId, evidence })
                ]
            );
            await this._recordEvent({
                assetId: outputId,
                transactionId,
                eventType: 'resource_produced',
                actorAccountId: actorId,
                details: { producerAssetId: producerId, originTileId: tile.tileId, quantity, unit: rule.output_unit }
            });
            await this._run(
                `INSERT INTO bank_production_claim
                 (id, transaction_id, idempotency_key, source_event_id, rule_id, producer_asset_id,
                  owner_account_id, origin_tile_id, elapsed_seconds, output_asset_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                    uuidv4(), transactionId, key, sourceId, normalizedRuleId, producerId,
                    ownerId, tile.tileId, elapsedSeconds, outputId
                ]
            );
            return { transaction: await this.getTransaction(transactionId), output: await this.getAsset(outputId) };
        });
    }

    async registerRecipe({ recipeName, processorAssetType, inputs, outputs, metadata = {}, id = uuidv4() }) {
        const recipeId = requireUuid(id, 'id');
        const name = requireString(recipeName, 'recipeName');
        const processorType = requireString(processorAssetType, 'processorAssetType');
        const normalizedInputs = normalizeRecipeMaterials(inputs, 'inputs');
        const normalizedOutputs = normalizeRecipeMaterials(outputs, 'outputs');
        return this._write(async () => {
            await this._run(
                `INSERT INTO bank_recipe
                 (id, recipe_name, processor_asset_type, inputs_json, outputs_json, metadata_json)
                 VALUES (?, ?, ?, ?, ?, ?)`,
                [
                    recipeId, name, processorType, JSON.stringify(normalizedInputs),
                    JSON.stringify(normalizedOutputs), JSON.stringify(metadata)
                ]
            );
            return {
                id: recipeId,
                recipeName: name,
                processorAssetType: processorType,
                inputs: normalizedInputs,
                outputs: normalizedOutputs,
                metadata
            };
        });
    }

    async transformResources({
        recipeId,
        processorAssetId,
        ownerAccountId,
        originTileId,
        inputAssetIds,
        batches = 1,
        sourceEventId,
        actorAccountId = ownerAccountId,
        idempotencyKey,
        evidence = {}
    }) {
        const normalizedRecipeId = requireUuid(recipeId, 'recipeId');
        const processorId = requireUuid(processorAssetId, 'processorAssetId');
        const ownerId = await this._requireAccount(ownerAccountId, 'ownerAccountId');
        const actorId = await this._requireAccount(actorAccountId, 'actorAccountId');
        const tile = parseTileId(originTileId);
        const key = requireString(idempotencyKey, 'idempotencyKey');
        const sourceId = requireString(sourceEventId, 'sourceEventId');
        if (!Number.isFinite(batches) || batches <= 0) throw new BankError('batches must be positive', 'INVALID_BATCHES');
        if (!Array.isArray(inputAssetIds) || inputAssetIds.length === 0 || inputAssetIds.length > 100) {
            throw new BankError('inputAssetIds must contain between 1 and 100 UUIDs', 'INVALID_ASSET_LIST');
        }
        const normalizedInputIds = [...new Set(inputAssetIds.map((id) => requireUuid(id, 'inputAssetId')))];
        if (normalizedInputIds.length !== inputAssetIds.length) {
            throw new BankError('inputAssetIds cannot contain duplicates', 'DUPLICATE_ASSET');
        }

        return this._write(async () => {
            const previous = await this._get(
                `SELECT transaction_id, output_asset_ids_json FROM bank_transformation WHERE idempotency_key = ?`,
                [key]
            );
            if (previous) {
                const outputIds = parseJson(previous.output_asset_ids_json);
                return {
                    transaction: await this.getTransaction(previous.transaction_id),
                    outputs: await Promise.all(outputIds.map((id) => this.getAsset(id)))
                };
            }
            const replay = await this._get(
                `SELECT id FROM bank_transformation WHERE source_event_id = ?`,
                [sourceId]
            );
            if (replay) throw new BankError('Transformation source event was already claimed', 'SOURCE_EVENT_REPLAY', 409);
            const recipe = await this._get('SELECT * FROM bank_recipe WHERE id = ? AND is_active = 1', [normalizedRecipeId]);
            if (!recipe) throw new BankError('Active recipe not found', 'RECIPE_NOT_FOUND', 404);
            const processor = await this._get('SELECT * FROM bank_asset WHERE id = ?', [processorId]);
            if (!processor || processor.owner_account_id !== ownerId) {
                throw new BankError('Processor owner mismatch', 'OWNER_MISMATCH', 409);
            }
            if (processor.asset_type !== recipe.processor_asset_type
                || processor.status !== 'deployed'
                || processor.location_tile_id !== tile.tileId) {
                throw new BankError('Required processor is not deployed on this parcel', 'INVALID_PROCESSOR', 409);
            }
            const parcel = await this._get(
                `SELECT a.owner_account_id FROM bank_parcel p JOIN bank_asset a ON a.id = p.title_asset_id
                 WHERE p.tile_id = ?`,
                [tile.tileId]
            );
            if (!parcel || parcel.owner_account_id !== ownerId) {
                throw new BankError('Transformation owner must own the parcel', 'PARCEL_OWNER_MISMATCH', 409);
            }
            const inputs = [];
            const totals = new Map();
            for (const assetId of normalizedInputIds) {
                const asset = await this._get('SELECT * FROM bank_asset WHERE id = ?', [assetId]);
                if (!asset) throw new BankError(`Input asset not found: ${assetId}`, 'ASSET_NOT_FOUND', 404);
                if (asset.kind !== 'resource_lot' || asset.owner_account_id !== ownerId
                    || asset.status !== 'active' || asset.custodian_account_id) {
                    throw new BankError(`Input lot is not available: ${assetId}`, 'INPUT_NOT_AVAILABLE', 409);
                }
                const materialKey = `${asset.asset_type}\u0000${asset.unit}`;
                totals.set(materialKey, (totals.get(materialKey) || 0) + asset.quantity);
                inputs.push(asset);
            }
            const requirements = parseJson(recipe.inputs_json);
            if (totals.size !== requirements.length) {
                throw new BankError('Input lots do not exactly match recipe materials', 'RECIPE_INPUT_MISMATCH', 409);
            }
            for (const requirement of requirements) {
                const materialKey = `${requirement.assetType}\u0000${requirement.unit}`;
                const required = requirement.quantity * batches;
                const supplied = totals.get(materialKey);
                if (!Number.isFinite(supplied) || Math.abs(supplied - required) > Math.max(1e-9, required * 1e-9)) {
                    throw new BankError('Input quantities do not conserve recipe requirements', 'RECIPE_INPUT_MISMATCH', 409);
                }
            }

            const transactionId = uuidv4();
            const outputIds = [];
            await this._recordTransaction({
                transactionId,
                transactionType: 'resource_transformation',
                actorAccountId: actorId,
                idempotencyKey: key,
                details: {
                    recipeId: normalizedRecipeId,
                    processorAssetId: processorId,
                    ownerAccountId: ownerId,
                    originTileId: tile.tileId,
                    batches,
                    sourceEventId: sourceId,
                    inputAssetIds: normalizedInputIds,
                    evidence
                }
            });
            for (const input of inputs) {
                await this._run(
                    `UPDATE bank_asset SET status = 'consumed', version = version + 1,
                     updated_at = CURRENT_TIMESTAMP WHERE id = ?`,
                    [input.id]
                );
                await this._recordEvent({
                    assetId: input.id,
                    transactionId,
                    eventType: 'resource_consumed_by_recipe',
                    actorAccountId: actorId,
                    details: { recipeId: normalizedRecipeId, processorAssetId: processorId }
                });
            }
            const outputs = parseJson(recipe.outputs_json);
            for (const output of outputs) {
                const outputId = uuidv4();
                outputIds.push(outputId);
                const outputQuantity = output.quantity * batches;
                await this._run(
                    `INSERT INTO bank_asset
                     (id, kind, asset_type, owner_account_id, quantity, unit, status, origin_tile_id,
                      location_tile_id, metadata_json)
                     VALUES (?, 'resource_lot', ?, ?, ?, ?, 'active', ?, ?, ?)`,
                    [
                        outputId, output.assetType, ownerId, outputQuantity, output.unit,
                        tile.tileId, tile.tileId,
                        JSON.stringify({ recipeId: normalizedRecipeId, processorAssetId: processorId, evidence })
                    ]
                );
                for (const input of inputs) {
                    await this._run(
                        `INSERT INTO bank_asset_lineage (parent_asset_id, child_asset_id, quantity)
                         VALUES (?, ?, NULL)`,
                        [input.id, outputId]
                    );
                }
                await this._recordEvent({
                    assetId: outputId,
                    transactionId,
                    eventType: 'resource_created_by_recipe',
                    actorAccountId: actorId,
                    details: {
                        recipeId: normalizedRecipeId,
                        processorAssetId: processorId,
                        inputAssetIds: normalizedInputIds,
                        quantity: outputQuantity,
                        unit: output.unit
                    }
                });
            }
            await this._run(
                `INSERT INTO bank_transformation
                 (id, transaction_id, idempotency_key, source_event_id, recipe_id, processor_asset_id,
                  owner_account_id, origin_tile_id, batches, input_asset_ids_json, output_asset_ids_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                    uuidv4(), transactionId, key, sourceId, normalizedRecipeId, processorId, ownerId,
                    tile.tileId, batches, JSON.stringify(normalizedInputIds), JSON.stringify(outputIds)
                ]
            );
            return {
                transaction: await this.getTransaction(transactionId),
                inputs: await Promise.all(normalizedInputIds.map((id) => this.getAsset(id))),
                outputs: await Promise.all(outputIds.map((id) => this.getAsset(id)))
            };
        });
    }

    async getProvenance(assetId) {
        const asset = await this.getAsset(assetId);
        const visited = new Set();
        const assets = [];
        const events = [];
        const lineage = [];
        const queue = [asset.id];

        while (queue.length > 0) {
            const currentId = queue.shift();
            if (visited.has(currentId)) continue;
            visited.add(currentId);
            assets.push(await this.getAsset(currentId));

            const currentEvents = await this._all(
                `SELECT * FROM bank_asset_event WHERE asset_id = ? ORDER BY created_at, id`,
                [currentId]
            );
            events.push(...currentEvents.map((row) => ({
                id: row.id,
                assetId: row.asset_id,
                transactionId: row.transaction_id,
                eventType: row.event_type,
                actorAccountId: row.actor_account_id,
                details: parseJson(row.details_json),
                createdAt: row.created_at
            })));

            const parents = await this._all(
                `SELECT * FROM bank_asset_lineage WHERE child_asset_id = ?`,
                [currentId]
            );
            for (const row of parents) {
                lineage.push({
                    parentAssetId: row.parent_asset_id,
                    childAssetId: row.child_asset_id,
                    quantity: row.quantity,
                    createdAt: row.created_at
                });
                queue.push(row.parent_asset_id);
            }
        }

        return { requestedAssetId: asset.id, assets, events, lineage };
    }

    async getPortfolio(accountId) {
        const account = await this.getAccount(accountId);
        const ownedRows = await this._all(
            `SELECT a.*, o.display_name AS owner_name, c.display_name AS custodian_name
             FROM bank_asset a
             JOIN bank_account o ON o.id = a.owner_account_id
             LEFT JOIN bank_account c ON c.id = a.custodian_account_id
             WHERE a.owner_account_id = ? AND a.status <> 'consumed'
             ORDER BY a.created_at, a.id`,
            [account.id]
        );
        const custodyRows = await this._all(
            `SELECT a.*, o.display_name AS owner_name, c.display_name AS custodian_name
             FROM bank_asset a
             JOIN bank_account o ON o.id = a.owner_account_id
             LEFT JOIN bank_account c ON c.id = a.custodian_account_id
             WHERE a.custodian_account_id = ? AND a.owner_account_id <> ? AND a.status <> 'consumed'
             ORDER BY a.created_at, a.id`,
            [account.id, account.id]
        );
        const parcels = await this._all(
            `SELECT p.tile_id, p.title_asset_id, p.sale_status, p.price_amount,
                    p.price_currency, p.metadata_json
             FROM bank_parcel p
             JOIN bank_asset a ON a.id = p.title_asset_id
             WHERE a.owner_account_id = ?
             ORDER BY p.tile_id`,
            [account.id]
        );
        return {
            account,
            ownedAssets: ownedRows.map(mapAsset),
            assetsInCustody: custodyRows.map(mapAsset),
            parcels: parcels.map((row) => ({
                tileId: row.tile_id,
                titleAssetId: row.title_asset_id,
                saleStatus: row.sale_status,
                priceAmount: row.price_amount,
                priceCurrency: row.price_currency,
                metadata: parseJson(row.metadata_json)
            }))
        };
    }
}

module.exports = {
    ACTIVE_ASSET_STATES,
    ASSET_KINDS,
    BankError,
    BankService,
    parseTileId
};
