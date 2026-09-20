const SERIALIZED_ASSET_KINDS = new Set([
    'vehicle', 'equipment', 'machinery', 'powerplant', 'warehouse', 'structure', 'land_title'
]);

const AUTHORITY = Object.freeze({
    identity: 'atlantis_x_user',
    money: 'greenland_bank',
    ownership: 'greenland_bank',
    custody: 'greenland_bank',
    provenance: 'greenland_bank',
    simulation: 'greenland_world',
    rendering: 'atlantis_terrain_webgpu'
});

function assetIdentityPolicy({ kind, quantity, unit }) {
    if (kind === 'resource_lot') {
        if (!Number.isFinite(quantity) || quantity <= 0 || !unit) {
            throw new Error('A resource lot UUID must identify a positive quantity and unit');
        }
        return 'one_uuid_per_conserved_lot';
    }
    if (!SERIALIZED_ASSET_KINDS.has(kind)) throw new Error(`Unsupported asset kind: ${kind}`);
    if (quantity != null || unit != null) {
        throw new Error('Serialized asset UUIDs identify one instance and do not carry fungible quantity');
    }
    return 'one_uuid_per_instance';
}

module.exports = { AUTHORITY, SERIALIZED_ASSET_KINDS, assetIdentityPolicy };
