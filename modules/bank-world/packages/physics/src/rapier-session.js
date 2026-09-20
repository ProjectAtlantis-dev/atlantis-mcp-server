const { LocalGeodeticFrame } = require('./local-frame');

class RapierSession {
    static async create({ RAPIER, origin, gravity = { x: 0, y: -9.81, z: 0 } }) {
        if (!RAPIER) throw new Error('Inject the Rapier module; the physics package does not hide its version');
        if (typeof RAPIER.init === 'function') await RAPIER.init();
        return new RapierSession({ RAPIER, origin, gravity });
    }

    constructor({ RAPIER, origin, gravity }) {
        this.RAPIER = RAPIER;
        this.frame = new LocalGeodeticFrame(origin);
        this.world = new RAPIER.World(gravity);
    }

    addTerrainHeightfield({ rows, columns, heights, scale, translation = { x: 0, y: 0, z: 0 } }) {
        if (heights.length !== (rows + 1) * (columns + 1)) {
            throw new Error('Rapier heightfield requires (rows + 1) * (columns + 1) heights');
        }
        const descriptor = this.RAPIER.ColliderDesc.heightfield(rows, columns, heights, scale)
            .setTranslation(translation.x, translation.y, translation.z);
        return this.world.createCollider(descriptor);
    }

    step(eventQueue) {
        if (eventQueue) this.world.step(eventQueue);
        else this.world.step();
    }
}

module.exports = { RapierSession };
