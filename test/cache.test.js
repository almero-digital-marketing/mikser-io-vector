// An embedding is the one thing here that a rebuild cannot reproduce for free.
//
// The vec0 index, the ids table and the mapped data all live in
// `mikser.sqlite`, which the engine unlinks on `--clear`, on a config edit and
// on a schema fingerprint that moved — correctly, because files are the source
// of truth. The embeddings used to go with it, so editing one line of config
// bought 11,000 API calls to get back exactly the vectors the site already had.
//
// These cover the cache that now stands in front of that: the round trip, the
// keys that must and must not collide, where the file is allowed to live, and
// the one that matters — a second plugin instance against a BRAND NEW engine
// database embedding nothing at all.

import { describe, it, before, after, beforeEach } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import * as sqliteVec from 'sqlite-vec'
import { createSqliteDatabase } from 'mikser-io/src/database/index.js'

import { vector } from '../index.js'
import { createEmbeddingCache, embeddingKey, resolveCachePath, DEFAULT_CACHE_FILENAME } from '../src/cache.js'

const DIM = 4

let dir
before(async () => { dir = await mkdtemp(path.join(tmpdir(), 'mikser-vector-cache-')) })
after(async () => { await rm(dir, { recursive: true, force: true }) })

const cacheIn = (name) => createEmbeddingCache({ filename: path.join(dir, name) })

describe('embeddingKey', () => {
    const base = { provider: 'openai', model: 'text-embedding-3-small', dim: 1536, text: 'hello' }

    it('is the same for the same text and model', () => {
        assert.equal(embeddingKey(base), embeddingKey({ ...base }))
    })

    it('separates two models, which answer the same text incomparably', () => {
        assert.notEqual(embeddingKey(base), embeddingKey({ ...base, model: 'text-embedding-3-large' }))
        assert.notEqual(embeddingKey(base), embeddingKey({ ...base, provider: 'mistral' }))
    })

    it('separates two widths, so a provider that silently rewidened a model misses', () => {
        assert.notEqual(embeddingKey(base), embeddingKey({ ...base, dim: 3072 }))
    })

    it('does not let a field boundary be forged out of the text', () => {
        // Joined on NUL rather than a printable separator: with '|' a model
        // named `a` and text `b|c` would key the same as model `a|b`, text `c`.
        assert.notEqual(
            embeddingKey({ provider: 'p', model: 'a', dim: 4, text: 'b' }),
            embeddingKey({ provider: 'p', model: 'a\u0000b', dim: 4, text: '' }))
    })
})

describe('the embedding cache', () => {
    it('gives back the floats it was given', () => {
        const cache = cacheIn('round-trip.sqlite')
        const vector = new Float32Array([0.25, -0.5, 0.75, 1])
        cache.set('k', vector, { provider: 'p', model: 'm', dim: DIM })
        assert.deepEqual([...cache.get('k', DIM)], [...vector])
        cache.close()
    })

    it('answers null for a key it has never seen', () => {
        const cache = cacheIn('miss.sqlite')
        assert.equal(cache.get('nothing', DIM), null)
        cache.close()
    })

    it('treats a stored width that does not match as a miss, not an error', () => {
        // The only way to get one is a provider changing a model's output
        // width without changing its name. Embedding again and overwriting is
        // the honest answer; failing the build is not.
        const cache = cacheIn('width.sqlite')
        cache.set('k', new Float32Array([1, 2, 3, 4]), { provider: 'p', model: 'm', dim: DIM })
        assert.equal(cache.get('k', 8), null)
        cache.close()
    })

    it('overwrites a key rather than accumulating rows for it', () => {
        const cache = cacheIn('overwrite.sqlite')
        cache.set('k', new Float32Array([1, 0, 0, 0]), { provider: 'p', model: 'm', dim: DIM })
        cache.set('k', new Float32Array([0, 1, 0, 0]), { provider: 'p', model: 'm', dim: DIM })
        assert.equal(cache.count(), 1)
        assert.deepEqual([...cache.get('k', DIM)], [0, 1, 0, 0])
        cache.close()
    })

    it('survives the process that wrote it', () => {
        // The whole point. The engine's cache is a file that gets unlinked;
        // this one is not, and a later run must find what the earlier one paid
        // for.
        const first = cacheIn('persist.sqlite')
        first.set('k', new Float32Array([1, 0, 0, 0]), { provider: 'p', model: 'm', dim: DIM })
        first.close()

        const second = cacheIn('persist.sqlite')
        assert.deepEqual([...second.get('k', DIM)], [1, 0, 0, 0])
        second.close()
    })

    it('reports how many it threw away when it is cleared', () => {
        const cache = cacheIn('clear.sqlite')
        for (const key of ['a', 'b', 'c']) {
            cache.set(key, new Float32Array([1, 0, 0, 0]), { provider: 'p', model: 'm', dim: DIM })
        }
        assert.equal(cache.clear(), 3, 'the count is the only evidence an irreversible thing was asked for')
        assert.equal(cache.count(), 0)
        cache.close()
    })
})

describe('resolveCachePath', () => {
    it('defaults to the runtime folder, which --clear keeps sqlite files in', () => {
        assert.equal(resolveCachePath({ runtimeFolder: '/w/runtime' }),
            path.join('/w/runtime', DEFAULT_CACHE_FILENAME))
    })

    it('refuses a name inside the runtime folder that --clear would delete', () => {
        // `--clear` empties the runtime folder except for .sqlite, .sqlite-wal
        // and .sqlite-shm. A cache named `vectors.db` would be deleted by the
        // one command this file exists to survive, and nothing would say so.
        assert.throws(() => resolveCachePath({ cache: 'vectors.db', runtimeFolder: '/w/runtime' }),
            /must end in \.sqlite/)
    })

    it('leaves a path outside the runtime folder alone', () => {
        assert.equal(resolveCachePath({ cache: '/elsewhere/vectors.db', runtimeFolder: '/w/runtime' }),
            '/elsewhere/vectors.db')
    })
})

// A model that answers deterministically and counts what it was asked.
function countingModel(calls) {
    return {
        specificationVersion: 'v2',
        provider: 'test',
        modelId: 'counter',
        maxEmbeddingsPerCall: 1,
        supportsParallelCalls: false,
        async doEmbed({ values }) {
            calls.push(values[0])
            return { embeddings: values.map(() => [1, 0, 0, 0]) }
        },
    }
}

// One plugin instance over one engine database. `runtimeFolder` is shared
// between instances on purpose: that is what makes the cache outlive the
// database next door.
async function runCycle({ runtimeFolder, databaseName, entity, calls, clearVectors = false }) {
    const db = createSqliteDatabase({
        runtimeFolder,
        version: 'test',
        config: { filename: databaseName },
        schemas: new Map([['mikser_entities',
            'CREATE TABLE IF NOT EXISTS mikser_entities (id TEXT PRIMARY KEY, data TEXT)']]),
        provisioners: [ctx => sqliteVec.load(ctx.handle)],
    })
    db.open()
    db.prepare('INSERT OR IGNORE INTO mikser_entities (id, data) VALUES (?, ?)').run(entity.id, '{}')

    const loaded = []
    let beforeRender
    const runtime = { options: { runtimeFolder, clearVectors } }
    const logged = []
    const logger = {
        error: (...a) => logged.push(a), warn: (...a) => logged.push(a), info: (...a) => logged.push(a),
        debug: () => {}, trace: () => {}, notice: () => {},
    }

    vector({ model: countingModel(calls), stores: { documents: { pick: ['content'] } } })({
        runtime,
        onLoaded: fn => loaded.push(fn),
        onBeforeRender: fn => { beforeRender = fn },
        onProvision: () => {},
        useLogger: () => logger,
        useJournal: () => [{ operation: 'create', entity }],
        useDatabase: () => db,
        registerSchema: (_name, sql) => db.exec(sql),
        registerRoute: () => {},
        registerTool: () => {},
        constants: { OPERATION: { CREATE: 'create', UPDATE: 'update', DELETE: 'delete' } },
    })

    for (const fn of loaded) await fn()
    await beforeRender()

    const indexed = db.prepare('SELECT count(*) AS n FROM mikser_vector_documents_ids').get().n
    db.close()
    return { indexed, runtime, logged }
}

describe('indexing against a wiped engine database', () => {
    const entity = { id: '/documents/a.md', type: 'document', content: 'the quick brown fox' }
    // What the model is actually asked for: `pick` builds an object and the
    // plugin TOON-encodes it, so the key is keyed on the encoding, not on the
    // raw field.
    const EMBEDDED = 'content: the quick brown fox'

    it('embeds once, and never again for the same text and model', async () => {
        const folder = await mkdtemp(path.join(dir, 'site-'))
        const calls = []

        const first = await runCycle({
            runtimeFolder: folder, databaseName: 'engine-1.sqlite', entity, calls,
        })
        assert.equal(first.indexed, 1, 'the entity should be in the index')
        // One probe at startup to learn the width, one for the entity.
        assert.deepEqual(calls, ['mikser', EMBEDDED])

        // A BRAND NEW engine database: what `--clear`, a config edit, or a
        // schema fingerprint bump leaves behind. Everything derived is gone
        // and has to be built again.
        calls.length = 0
        const second = await runCycle({
            runtimeFolder: folder, databaseName: 'engine-2.sqlite', entity, calls,
        })
        assert.equal(second.indexed, 1, 'the index must be rebuilt')
        assert.deepEqual(calls, ['mikser'],
            'the entity was embedded again — that is one paid API call per entity on every config edit')
    })

    it('embeds again once --clear-vectors has emptied the cache', async () => {
        const folder = await mkdtemp(path.join(dir, 'cleared-'))
        const calls = []
        await runCycle({ runtimeFolder: folder, databaseName: 'engine-1.sqlite', entity, calls })

        calls.length = 0
        const second = await runCycle({
            runtimeFolder: folder, databaseName: 'engine-2.sqlite', entity, calls,
            // Delivered on runtime.options, which is how both the CLI parse
            // and a forwarded request hand a plugin option over.
            clearVectors: true,
        })
        assert.equal(second.indexed, 1)
        assert.deepEqual(calls, ['mikser', EMBEDDED],
            'the cache was emptied, so this entity has to be paid for again')
    })

    it('consumes the flag, so a watcher empties the cache once', async () => {
        const folder = await mkdtemp(path.join(dir, 'consumed-'))
        const calls = []
        const run = await runCycle({
            runtimeFolder: folder, databaseName: 'engine-1.sqlite', entity, calls,
            clearVectors: true,
        })
        assert.equal(run.runtime.options.clearVectors, false,
            'the flag outlived the cycle that acted on it, so every later save would empty the cache too')
    })
})
