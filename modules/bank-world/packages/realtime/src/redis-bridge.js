const { EventEmitter } = require('node:events');
const { validateEnvelope } = require('../../protocol/src');

const RELEASE_LEASE_SCRIPT = `
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
`;

class RedisRealtimeBridge extends EventEmitter {
    constructor({ publisher, subscriber, namespace = 'greenland:test:', streamMaxLength = 2048 }) {
        super();
        if (!publisher || !subscriber) throw new Error('Dedicated Redis publisher and subscriber clients are required');
        if (!namespace.startsWith('greenland:test:')) throw new Error('Redis namespace must start with greenland:test:');
        this.publisher = publisher;
        this.subscriber = subscriber;
        this.namespace = namespace;
        this.streamMaxLength = streamMaxLength;
        this.handlers = new Map();
        this.started = false;
        this.onMessage = this.onMessage.bind(this);
    }

    channel(roomId) { return `${this.namespace}room:${roomId}:live`; }
    stream(roomId) { return `${this.namespace}room:${roomId}:stream`; }
    presenceKey(accountId) { return `${this.namespace}presence:${accountId}`; }
    leaseKey(resourceId) { return `${this.namespace}lease:${resourceId}`; }

    async start() {
        if (this.started) return;
        this.started = true;
        this.subscriber.on('message', this.onMessage);
    }

    async stop() {
        if (!this.started) return;
        const channels = [...this.handlers.keys()];
        if (channels.length) await this.subscriber.unsubscribe(...channels);
        this.subscriber.off('message', this.onMessage);
        this.handlers.clear();
        this.started = false;
    }

    onMessage(channel, encoded) {
        const handler = this.handlers.get(channel);
        if (!handler) return;
        try {
            handler(validateEnvelope(JSON.parse(encoded)));
        } catch (error) {
            this.emit('invalidMessage', { channel, error });
        }
    }

    async subscribe(roomId, handler) {
        if (!this.started) await this.start();
        const channel = this.channel(roomId);
        this.handlers.set(channel, handler);
        await this.subscriber.subscribe(channel);
        return async () => {
            this.handlers.delete(channel);
            await this.subscriber.unsubscribe(channel);
        };
    }

    async publish(roomId, message) {
        const envelope = validateEnvelope(message);
        if (envelope.roomId !== roomId) {
            throw new Error('Redis roomId must match the protocol envelope roomId');
        }
        const encoded = JSON.stringify(envelope);
        const streamId = await this.publisher.xadd(
            this.stream(roomId),
            'MAXLEN', '~', this.streamMaxLength,
            '*', 'message', encoded
        );
        await this.publisher.publish(this.channel(roomId), encoded);
        return { streamId, envelope };
    }

    async replay(roomId, afterStreamId = '0-0', count = 256) {
        const response = await this.publisher.xread(
            'COUNT', count,
            'STREAMS', this.stream(roomId), afterStreamId
        );
        if (!response) return [];
        return response.flatMap(([_stream, entries]) => entries.map(([streamId, fields]) => {
            const messageIndex = fields.indexOf('message');
            if (messageIndex === -1) throw new Error('Redis room stream entry has no message field');
            return { streamId, envelope: validateEnvelope(JSON.parse(fields[messageIndex + 1])) };
        }));
    }

    async setPresence({ accountId, serverId, roomId, ttlMs = 30000 }) {
        if (!accountId || !serverId || !roomId) throw new Error('accountId, serverId, and roomId are required');
        const value = JSON.stringify({ accountId, serverId, roomId, observedAt: new Date().toISOString() });
        await this.publisher.set(this.presenceKey(accountId), value, 'PX', ttlMs);
        return JSON.parse(value);
    }

    async acquireLease({ resourceId, holderId, ttlMs = 10000 }) {
        const result = await this.publisher.set(this.leaseKey(resourceId), holderId, 'PX', ttlMs, 'NX');
        return result === 'OK';
    }

    async releaseLease({ resourceId, holderId }) {
        return (await this.publisher.eval(
            RELEASE_LEASE_SCRIPT,
            1,
            this.leaseKey(resourceId),
            holderId
        )) === 1;
    }
}

module.exports = { RedisRealtimeBridge, RELEASE_LEASE_SCRIPT };
