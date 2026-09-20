const fs = require('node:fs');
const path = require('node:path');
const express = require('express');
const cors = require('cors');
const { BankService } = require('../../../packages/bank/src/bank-service');
const { createBankRouter } = require('../../../packages/bank/src/bank-routes');
const { GameService } = require('../../../packages/world/src/game-service');
const { createGameRouter } = require('../../../packages/world/src/game-routes');
const { assertTestOnlyRuntime } = require('../../../packages/persistence/src/test-config');
const { createWorldLoop } = require('../../../packages/world/src/world-loop');
const { createWorldRelay } = require('../../../packages/realtime/src/world-relay');

function createRuntime(env = process.env) {
    const repositoryRoot = path.resolve(__dirname, '../../..');
    const requestedBankPath = env.BANK_DB_PATH || path.join(repositoryRoot, '.local/greenland_bank.test.sqlite');
    const requestedWorldPath = env.WORLD_DB_PATH || path.join(repositoryRoot, '.local/greenland_world.test.sqlite');
    const paths = assertTestOnlyRuntime({
        env,
        bankDbPath: requestedBankPath,
        worldDbPath: requestedWorldPath
    });
    fs.mkdirSync(path.dirname(paths.bankDbPath), { recursive: true });
    fs.mkdirSync(path.dirname(paths.worldDbPath), { recursive: true });

    const bank = new BankService({ dbPath: paths.bankDbPath });
    const bankReady = bank.open();
    const world = new GameService({ dbPath: paths.worldDbPath, bank });
    const worldReady = bankReady.then(() => world.open());
    return { bank, bankReady, world, worldReady, paths };
}

function createApplication({ env = process.env, runtime = createRuntime(env) } = {}) {
    const app = express();
    app.use(cors());
    app.use(express.json({ limit: '256kb' }));
    app.use('/api/bank', createBankRouter({
        bank: runtime.bank,
        ready: runtime.bankReady,
        authorityToken: env.BANK_AUTHORITY_TOKEN || ''
    }));
    app.use('/api/game', createGameRouter({
        game: runtime.world,
        ready: runtime.worldReady,
        authorityToken: env.GAME_SERVER_AUTHORITY_TOKEN || ''
    }));
    app.get('/health', async (_req, res) => {
        await Promise.all([runtime.bankReady, runtime.worldReady]);
        res.json({
            status: 'OK',
            environment: 'test-only',
            rendererTarget: 'webgpu',
            bankMutationsEnabled: Boolean(env.BANK_AUTHORITY_TOKEN),
            gameCommandsEnabled: Boolean(env.GAME_SERVER_AUTHORITY_TOKEN)
        });
    });
    return { app, runtime };
}

async function start(env = process.env) {
    const { app, runtime } = createApplication({ env });
    await Promise.all([runtime.bankReady, runtime.worldReady]);
    let relay;
    try {
        if(env.GREENLAND_REDIS_TEST_URL || env.GREENLAND_REDIS_NAMESPACE || env.GREENLAND_WORLD_ROOM_ID) {
            relay=await createWorldRelay(env);
        }
    } catch(error) {
        await runtime.world.close();await runtime.bank.close();throw error;
    }
    const port = Number(env.PORT || 3010);
    let server;
    try {
        server = await new Promise((resolve,reject)=>{
            const listener=app.listen(port,'127.0.0.1',()=>resolve(listener));
            listener.once('error',reject);
        });
    } catch(error) {
        if(relay)await relay.close();
        await runtime.world.close();await runtime.bank.close();throw error;
    }
    console.log(`Greenland test game API listening on http://127.0.0.1:${server.address().port}`);
    const loop = createWorldLoop({world:runtime.world,
      afterStep:relay ? async()=>relay.publish(await runtime.world.getWorldState()) : null,
      onError:error=>{
        console.error('Authoritative world loop failed:',error);
        process.exitCode=1;
        void close().catch(closeError=>{console.error('World shutdown failed:',closeError);});
    }});
    let closing;
    const close = () => closing ??= (async () => {
        await loop.stop();
        await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
        await runtime.world.close();
        await runtime.bank.close();
        if(relay) await relay.close();
    })();
    loop.start();
    return { server, close, runtime };
}

if (require.main === module) {
    start().then(({close})=>{
        for(const signal of ['SIGTERM','SIGINT']) process.once(signal,()=>{
            void close().catch(error=>{console.error(error);process.exitCode=1;});
        });
    }).catch((error) => {
        console.error(error.message);
        process.exitCode = 1;
    });
}

module.exports = { createApplication, createRuntime, start };
