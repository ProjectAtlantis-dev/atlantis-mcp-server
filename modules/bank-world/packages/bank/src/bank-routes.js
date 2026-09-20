const crypto = require('crypto');
const express = require('express');
const { BankError } = require('./bank-service');

function tokensMatch(actual, expected) {
    if (!actual || !expected) return false;
    const actualBuffer = Buffer.from(actual);
    const expectedBuffer = Buffer.from(expected);
    return actualBuffer.length === expectedBuffer.length
        && crypto.timingSafeEqual(actualBuffer, expectedBuffer);
}

function createBankRouter({ bank, ready, authorityToken }) {
    const router = express.Router();

    const handler = (fn) => async (req, res) => {
        try {
            await ready;
            await fn(req, res);
        } catch (error) {
            const status = error instanceof BankError ? error.status : 500;
            if (!(error instanceof BankError)) console.error('Bank API error:', error);
            res.status(status).json({
                success: false,
                error: error.message || 'Bank operation failed',
                code: error.code || 'INTERNAL_ERROR'
            });
        }
    };

    const requireAuthority = (req, res, next) => {
        if (!authorityToken) {
            return res.status(503).json({
                success: false,
                error: 'BANK_AUTHORITY_TOKEN is not configured; bank mutations are disabled',
                code: 'BANK_MUTATIONS_DISABLED'
            });
        }
        const authorization = req.get('authorization') || '';
        const token = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
        if (!tokensMatch(token, authorityToken)) {
            return res.status(401).json({
                success: false,
                error: 'Trusted bank authority token required',
                code: 'BANK_AUTHORITY_REQUIRED'
            });
        }
        next();
    };

    const requireService = async (req, res, next) => {
        try {
            await ready;
            const authorization = req.get('authorization') || '';
            const token = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
            req.bankServiceIdentity = await bank.authenticateServiceToken(token);
            next();
        } catch (error) {
            const status = error instanceof BankError ? error.status : 500;
            res.status(status).json({
                success: false,
                error: error.message || 'Service authentication failed',
                code: error.code || 'INTERNAL_ERROR'
            });
        }
    };

    const requireGameAuthorityService = (req, res, next) => {
        if (req.bankServiceIdentity?.serviceType !== 'game_authority') {
            return res.status(403).json({
                success: false,
                error: 'A registered game-authority service is required',
                code: 'GAME_AUTHORITY_SERVICE_REQUIRED'
            });
        }
        next();
    };

    router.get('/health', handler(async (_req, res) => {
        res.json({
            status: 'OK',
            authorityMutationsEnabled: !!authorityToken,
            parcelDepth: 12,
            resourceModel: 'conserved-lots'
        });
    }));

    router.get('/assets/:assetId/verify', handler(async (req, res) => {
        res.json(await bank.verifyAsset(req.params.assetId));
    }));

    router.get('/assets/:assetId/provenance', handler(async (req, res) => {
        res.json(await bank.getProvenance(req.params.assetId));
    }));

    router.get('/accounts/:accountId/portfolio', handler(async (req, res) => {
        res.json(await bank.getPortfolio(req.params.accountId));
    }));

    router.get('/accounts/:accountId/balance', handler(async (req, res) => {
        res.json(await bank.getBalance(req.params.accountId, req.query.currency || 'GLC'));
    }));

    router.get('/transactions/:transactionId', handler(async (req, res) => {
        res.json(await bank.getTransaction(req.params.transactionId));
    }));

    router.get('/warehouse/quotes/:quoteId', handler(async (req, res) => {
        res.json(await bank.getWarehouseQuote(req.params.quoteId));
    }));

    router.get('/parcels/:tileId', handler(async (req, res) => {
        res.json(await bank.getParcel(req.params.tileId));
    }));

    router.get('/parcels', handler(async (req, res) => {
        const tileIds = typeof req.query.tileIds === 'string'
            ? req.query.tileIds.split(',').map((value) => value.trim()).filter(Boolean)
            : [];
        res.json(await bank.getParcels(tileIds));
    }));

    router.post('/accounts', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await bank.createAccount(req.body));
    }));

    router.post('/accounts/resolve', requireAuthority, handler(async (req, res) => {
        res.json(await bank.resolveAccount(req.body));
    }));

    router.post('/services/register', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await bank.registerServiceIdentity(req.body));
    }));

    router.get('/services/me', requireService, handler(async (req, res) => {
        res.json(req.bankServiceIdentity);
    }));

    router.post('/parcels/register', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await bank.registerParcel(req.body));
    }));

    router.post('/parcels/purchase', requireAuthority, handler(async (req, res) => {
        res.json(await bank.purchaseParcels(req.body));
    }));

    router.post('/assets/issue', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await bank.issueAsset(req.body));
    }));

    router.post('/assets/:assetId/transfer', requireAuthority, handler(async (req, res) => {
        res.json(await bank.transferAsset({ ...req.body, assetId: req.params.assetId }));
    }));

    router.post('/assets/:assetId/custody', requireAuthority, handler(async (req, res) => {
        res.json(await bank.changeCustody({ ...req.body, assetId: req.params.assetId }));
    }));

    router.post('/assets/:assetId/deploy', requireAuthority, handler(async (req, res) => {
        res.json(await bank.deployAsset({ ...req.body, assetId: req.params.assetId }));
    }));

    router.post('/assets/:assetId/split', requireAuthority, handler(async (req, res) => {
        res.json(await bank.splitResourceLot({ ...req.body, assetId: req.params.assetId }));
    }));

    router.post('/credits/transfer', requireAuthority, handler(async (req, res) => {
        res.json(await bank.transferCredits(req.body));
    }));

    router.post('/production/rules', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await bank.registerProductionRule(req.body));
    }));

    router.post('/recipes', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await bank.registerRecipe(req.body));
    }));

    router.post(
        '/production/claims',
        requireService,
        requireGameAuthorityService,
        handler(async (req, res) => {
            res.status(201).json(await bank.claimProduction({
                ...req.body,
                evidence: {
                    ...(req.body.evidence || {}),
                    sourceServiceId: req.bankServiceIdentity.id,
                    sourceServiceName: req.bankServiceIdentity.serviceName
                }
            }));
        })
    );

    router.post(
        '/transformations',
        requireService,
        requireGameAuthorityService,
        handler(async (req, res) => {
            res.status(201).json(await bank.transformResources({
                ...req.body,
                evidence: {
                    ...(req.body.evidence || {}),
                    sourceServiceId: req.bankServiceIdentity.id,
                    sourceServiceName: req.bankServiceIdentity.serviceName
                }
            }));
        })
    );

    router.post('/warehouse/quotes', requireService, handler(async (req, res) => {
        if (req.bankServiceIdentity.serviceType !== 'warehouse') {
            throw new BankError('Only a warehouse service can register a warehouse quote', 'WAREHOUSE_SERVICE_REQUIRED', 403);
        }
        res.status(201).json(await bank.createWarehouseQuote({
            ...req.body,
            warehouseAccountId: req.bankServiceIdentity.ownerAccountId
        }));
    }));

    router.post('/warehouse/listings', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await bank.createWarehouseListing(req.body));
    }));

    router.post('/warehouse/quotes/:quoteId/settle', requireAuthority, handler(async (req, res) => {
        res.json(await bank.settleWarehouseQuote({ ...req.body, quoteId: req.params.quoteId }));
    }));

    return router;
}

module.exports = { createBankRouter };
