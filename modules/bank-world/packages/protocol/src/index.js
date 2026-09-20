const PROTOCOL_VERSION = 1;
const MESSAGE_KINDS = new Set(['command', 'input', 'snapshot', 'delta', 'event', 'ack', 'error']);

function assertSafeSequence(value, name) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${name} must be a non-negative safe integer`);
    return value;
}

function createEnvelope({ kind, roomId, sequence, tick, payload, correlationId = null }) {
    if (!MESSAGE_KINDS.has(kind)) throw new Error(`Unsupported message kind: ${kind}`);
    if (typeof roomId !== 'string' || !roomId.trim()) throw new Error('roomId is required');
    return {
        version: PROTOCOL_VERSION,
        kind,
        roomId: roomId.trim(),
        sequence: assertSafeSequence(sequence, 'sequence'),
        tick: assertSafeSequence(tick, 'tick'),
        correlationId,
        payload: payload ?? null
    };
}

function validateEnvelope(message) {
    if (!message || typeof message !== 'object') throw new Error('Protocol message must be an object');
    if (message.version !== PROTOCOL_VERSION) throw new Error(`Unsupported protocol version: ${message.version}`);
    return createEnvelope(message);
}

function uuidToBytes(uuid) {
    const hex = String(uuid).toLowerCase().replace(/-/g, '');
    if (!/^[0-9a-f]{32}$/.test(hex)) throw new Error('UUID is invalid');
    return Uint8Array.from(hex.match(/.{2}/g), (pair) => Number.parseInt(pair, 16));
}

function bytesToUuid(bytes) {
    if (!(bytes instanceof Uint8Array) || bytes.length !== 16) throw new Error('UUID bytes must be Uint8Array(16)');
    const hex = [...bytes].map((value) => value.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

module.exports = {
    MESSAGE_KINDS,
    PROTOCOL_VERSION,
    bytesToUuid,
    createEnvelope,
    uuidToBytes,
    validateEnvelope
};
