// Persistent single-host simulation bank. Deliberately does not start the old
// voyage/world loop: Terrain's simulation remains the only position authority.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const express = require('express');
const { BankService, BankError } = require('../../../packages/bank/src/bank-service');
const { createBankRouter } = require('../../../packages/bank/src/bank-routes');

async function start(env = process.env) {
    const dbPath = env.ATLANTIS_BANK_DB_PATH;
    const token = env.BANK_AUTHORITY_TOKEN;
    const port = Number(env.ATLANTIS_BANK_PORT);
    if (!dbPath || !path.isAbsolute(dbPath) || !dbPath.endsWith('.sqlite')) {
        throw new Error('ATLANTIS_BANK_DB_PATH must select an absolute persistent .sqlite file');
    }
    if (!token || token.length < 32) throw new Error('BANK_AUTHORITY_TOKEN must contain at least 32 characters');
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Explicit ATLANTIS_BANK_PORT required');
    if (env.BANK_DATABASE_URL || env.DATABASE_URL || env.WORLD_DATABASE_URL) {
        throw new Error('SQL connection URLs are not supported by the single-host SQLite bank');
    }
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    const bank = new BankService({ dbPath });
    await bank.open();
    const app = express();
    app.disable('x-powered-by');
    // Protect reads too. No browser CORS surface and no credentials in viewer URLs.
    app.use(async (req, res, next) => {
        const supplied = Buffer.from(req.get('authorization') || '');
        const expected = Buffer.from(`Bearer ${token}`);
        if (supplied.length !== expected.length || !crypto.timingSafeEqual(supplied, expected)) {
            try {
                const authorization = req.get('authorization') || '';
                await bank.authenticateServiceToken(authorization.startsWith('Bearer ') ? authorization.slice(7) : '');
            } catch (error) {
                if (!(error instanceof BankError)) return next(error);
                return res.status(401).json({ code: 'BANK_AUTHORITY_REQUIRED' });
            }
        }
        next();
    });
    app.use(express.json({ limit: '256kb' }));
    app.get('/health', (_req, res) => res.json({ status: 'OK',
        mode: 'persistent-single-host', positionAuthority: false, service: 'atlantis-bank' }));
    // The router checks per-service credentials for production/warehouse calls.
    // Passing the outer boundary does not grant mint/transfer permission: those
    // routes still require the authority token, not a warehouse service token.
    app.use('/api/bank', createBankRouter({ bank, ready: Promise.resolve(), authorityToken: token }));
    let server;
    try {
        server = await new Promise((resolve, reject) => {
            const listener = app.listen(port, '127.0.0.1', () => resolve(listener));
            listener.once('error', reject);
        });
    } catch (error) {
        await bank.close();
        throw error;
    }
    let closing;
    const close = () => closing ??= (async () => {
        await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
        await bank.writeQueue;
        await bank.close();
    })();
    return { server, bank, close };
}

if (require.main === module) {
    start().then(({ close }) => {
        console.log('Persistent Atlantis bank listening on loopback');
        for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => {
            close().catch(error => { console.error(error); process.exitCode = 1; });
        });
    }).catch(error => { console.error(error); process.exitCode = 1; });
}

module.exports = { start };
