const { EventEmitter } = require('node:events');
const { createEnvelope } = require('../../protocol/src');

class AuthoritativeRoom extends EventEmitter {
    constructor({ id, maxReplayMessages = 256 }) {
        super();
        if (!id) throw new Error('Room id is required');
        this.id = id;
        this.maxReplayMessages = maxReplayMessages;
        this.sequence = 0;
        this.tick = 0;
        this.replay = [];
    }

    advanceTick(count = 1) {
        if (!Number.isSafeInteger(count) || count < 1) throw new Error('count must be a positive integer');
        this.tick += count;
        return this.tick;
    }

    publish(kind, payload, correlationId = null) {
        const message = createEnvelope({
            kind,
            roomId: this.id,
            sequence: ++this.sequence,
            tick: this.tick,
            payload,
            correlationId
        });
        this.replay.push(message);
        if (this.replay.length > this.maxReplayMessages) this.replay.shift();
        this.emit('message', message);
        return message;
    }

    messagesAfter(sequence) {
        return this.replay.filter((message) => message.sequence > sequence);
    }
}

class RealtimeTransport {
    async start() { throw new Error('RealtimeTransport.start is not implemented'); }
    async stop() { throw new Error('RealtimeTransport.stop is not implemented'); }
    async publish(_roomId, _message) { throw new Error('RealtimeTransport.publish is not implemented'); }
}

class InMemoryRealtimeTransport extends RealtimeTransport {
    constructor() {
        super();
        this.rooms = new Map();
    }

    async start() {}
    async stop() { this.rooms.clear(); }

    room(roomId) {
        if (!this.rooms.has(roomId)) this.rooms.set(roomId, new AuthoritativeRoom({ id: roomId }));
        return this.rooms.get(roomId);
    }

    async publish(roomId, { kind, payload, correlationId }) {
        return this.room(roomId).publish(kind, payload, correlationId);
    }
}

module.exports = { AuthoritativeRoom, InMemoryRealtimeTransport, RealtimeTransport };
