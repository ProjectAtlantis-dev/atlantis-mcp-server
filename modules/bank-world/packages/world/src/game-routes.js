const crypto = require('crypto');
const express = require('express');
const { GameError } = require('./game-service');

function tokensMatch(actual, expected) {
    if (!actual || !expected) return false;
    const actualBuffer = Buffer.from(actual);
    const expectedBuffer = Buffer.from(expected);
    return actualBuffer.length === expectedBuffer.length
        && crypto.timingSafeEqual(actualBuffer, expectedBuffer);
}

function createGameRouter({ game, ready, authorityToken }) {
    const router = express.Router();

    const handler = (fn) => async (req, res) => {
        try {
            await ready;
            await fn(req, res);
        } catch (error) {
            const status = error instanceof GameError ? error.status : 500;
            if (!(error instanceof GameError)) console.error('Game API error:', error);
            res.status(status).json({
                success: false,
                error: error.message || 'Game operation failed',
                code: error.code || 'INTERNAL_ERROR'
            });
        }
    };

    const requireAuthority = (req, res, next) => {
        if (!authorityToken) {
            return res.status(503).json({
                success: false,
                error: 'GAME_SERVER_AUTHORITY_TOKEN is not configured; game commands are disabled',
                code: 'GAME_COMMANDS_DISABLED'
            });
        }
        const authorization = req.get('authorization') || '';
        const token = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
        if (!tokensMatch(token, authorityToken)) {
            return res.status(401).json({
                success: false,
                error: 'Trusted game authority token required',
                code: 'GAME_AUTHORITY_REQUIRED'
            });
        }
        next();
    };

    router.get('/health', handler(async (_req, res) => {
        res.json({
            status: 'OK',
            authorityCommandsEnabled: !!authorityToken,
            rendererTarget: 'webgpu',
            worldAuthority: 'server-timed'
        });
    }));

    router.get('/world', requireAuthority, handler(async (req, res) => {
        res.json(await game.getWorldState({ afterEventSeq: req.query.afterEventSeq }));
    }));

    router.post('/players/bootstrap', requireAuthority, handler(async (req, res) => {
        res.json(await game.bootstrapPlayer(req.body));
    }));

    router.get('/players/by-external/:externalUserId', requireAuthority, handler(async (req, res) => {
        res.json(await game.getPlayerState(req.params.externalUserId));
    }));

    router.post('/vehicles/register', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await game.registerVehicle(req.body));
    }));

    router.get('/vehicles/:assetId', requireAuthority, handler(async (req, res) => {
        res.json(await game.getVehicle(req.params.assetId));
    }));

    router.post('/voyages/quote', requireAuthority, handler(async (req, res) => {
        res.json(await game.quoteVoyage(req.body));
    }));

    router.post('/voyages/depart', requireAuthority, handler(async (req, res) => {
        res.status(201).json(await game.departVoyage(req.body));
    }));

    router.get('/voyages/:transitId', requireAuthority, handler(async (req, res) => {
        res.json(await game.getTransit(req.params.transitId));
    }));

    router.post('/world/advance', requireAuthority, handler(async (req, res) => {
        res.json(await game.advanceWorld({ now: req.body.now }));
    }));

    router.post('/cargo/load', requireAuthority, handler(async (req, res) => {
        res.json(await game.loadCargo(req.body));
    }));

    router.post('/cargo/unload', requireAuthority, handler(async (req, res) => {
        res.json(await game.unloadCargo(req.body));
    }));

    return router;
}

module.exports = { createGameRouter };
