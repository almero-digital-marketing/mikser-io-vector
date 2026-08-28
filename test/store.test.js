import { describe, it, before, beforeEach, after } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import * as sqliteVec from 'sqlite-vec'
import { createSqliteDatabase } from 'mikser-io/src/database/index.js'
import { createStore } from '../src/store.js'

// The store runs on the ENGINE's connection with the sqlite-vec extension
// loaded onto it — not on a database of its own. So these run against a real
// engine database with a real extension, provisioned exactly the way the plugin
// provisions it (`onProvision(ctx => sqliteVec.load(ctx.handle))`). A double
// would test the double: vec0 is a virtual table with its own rules about
// LIMIT, rowid binding and what a trigger can reach, and those rules are the
// thing worth checking.

const DIM = 4
const STORES = { pages: {}, media: {} }

// Unit vectors, so cosine distance is the natural metric and the expected
// ordering is arguable from the numbers rather than from running it.
const vec = (...xs) => {
    const norm = Math.hypot(...xs) || 1
    return new Float32Array(xs.map(x => x / norm))
}
const NORTH = vec(1, 0, 0, 0)
const NEAR_NORTH = vec(0.95, 0.31, 0, 0)
const EAST = vec(0, 1, 0, 0)

let dir, db, store

before(async () => {
    dir = await mkdtemp(path.join(tmpdir(), 'mikser-vector-'))
})

after(async () => {
    db?.close?.()
    await rm(dir, { recursive: true, force: true })
})

beforeEach(async () => {
    db?.close?.()
    db = createSqliteDatabase({
        runtimeFolder: dir,
        version: 'test',
        config: { filename: `vec-${Math.random().toString(36).slice(2)}.sqlite` },
        // The catalog table the ids table has its foreign key into. Without it
        // the FK is unenforceable and the cascade this file cares about cannot
        // be observed at all.
        schemas: new Map([['mikser_entities',
            'CREATE TABLE IF NOT EXISTS mikser_entities (id TEXT PRIMARY KEY, data TEXT)']]),
        provisioners: [ctx => sqliteVec.load(ctx.handle)],
    })
    db.open()

    // The store registers its schemas through registerSchema and relies on them
    // being applied immediately, because the engine database is already open by
    // the time its onLoaded runs. Applying directly is that same contract.
    store = await createStore({
        db, dim: DIM, stores: STORES,
        registerSchema: (_name, sql) => db.exec(sql),
    })
})

const seedEntity = (id) =>
    db.prepare('INSERT OR IGNORE INTO mikser_entities (id, data) VALUES (?, ?)').run(id, '{}')

describe('createStore', () => {
    it('refuses a database that is not open, rather than failing later on a prepare', async () => {
        await assert.rejects(
            () => createStore({ db: { isOpen: false }, dim: DIM, stores: STORES, registerSchema: () => {} }),
            /requires an open engine database handle/)
    })

    it('creates a vec0 table and an ids table per configured store', () => {
        const names = db.prepare(
            "SELECT name FROM sqlite_master WHERE name LIKE 'mikser_vector_%' ORDER BY name"
        ).all().map(r => r.name)
        for (const store of Object.keys(STORES)) {
            assert.ok(names.includes(`mikser_vector_${store}`), `${store} vec table`)
            assert.ok(names.includes(`mikser_vector_${store}_ids`), `${store} ids table`)
        }
    })
})

describe('upsert', () => {
    it('inserts, and finds what it inserted', async () => {
        seedEntity('/documents/a.md')
        await store.upsert('pages', '/documents/a.md', NORTH, { title: 'A' })
        const hits = await store.findSimilar('pages', NORTH, 5)
        assert.equal(hits.length, 1)
        assert.equal(hits[0].id, '/documents/a.md')
        assert.equal(hits[0].title, 'A')
    })

    it('UPDATES on a second call rather than adding a second row', async () => {
        // entity_id is UNIQUE, so a re-embed of the same entity must move the
        // vector, not accumulate one row per build.
        seedEntity('/documents/a.md')
        await store.upsert('pages', '/documents/a.md', NORTH, { title: 'first' })
        await store.upsert('pages', '/documents/a.md', EAST, { title: 'second' })

        const rows = db.prepare('SELECT count(*) AS n FROM mikser_vector_pages_ids').get()
        assert.equal(rows.n, 1)
        // Both halves moved: the data and the vector itself.
        const hits = await store.findSimilar('pages', EAST, 5)
        assert.equal(hits[0].title, 'second')
        assert.ok(hits[0].distance < 0.01, 'the stored vector should now be EAST')
    })

    it('keeps stores separate', async () => {
        seedEntity('/documents/a.md')
        await store.upsert('pages', '/documents/a.md', NORTH, { where: 'pages' })
        assert.equal((await store.findSimilar('media', NORTH, 5)).length, 0)
        assert.equal((await store.findSimilar('pages', NORTH, 5)).length, 1)
    })

    it('accepts an entity with no mapped data', async () => {
        seedEntity('/documents/a.md')
        await store.upsert('pages', '/documents/a.md', NORTH, null)
        const hits = await store.findSimilar('pages', NORTH, 5)
        assert.equal(hits[0].id, '/documents/a.md')
    })
})

describe('findSimilar', () => {
    beforeEach(async () => {
        for (const [id, v, data] of [
            ['/n',  NORTH,      { name: 'north' }],
            ['/nn', NEAR_NORTH, { name: 'near-north' }],
            ['/e',  EAST,       { name: 'east' }],
        ]) {
            seedEntity(id)
            await store.upsert('pages', id, v, data)
        }
    })

    it('orders by distance, nearest first', async () => {
        const hits = await store.findSimilar('pages', NORTH, 3)
        assert.deepEqual(hits.map(h => h.name), ['north', 'near-north', 'east'])
        // Monotonically non-decreasing, not merely "the first one is right".
        for (let i = 1; i < hits.length; i++) {
            assert.ok(hits[i].distance >= hits[i - 1].distance,
                `distance must not decrease: ${hits[i - 1].distance} then ${hits[i].distance}`)
        }
    })

    it('honours the limit, which vec0 requires on the MATCH query itself', async () => {
        assert.equal((await store.findSimilar('pages', NORTH, 1)).length, 1)
        assert.equal((await store.findSimilar('pages', NORTH, 2)).length, 2)
    })

    it('spreads mapped data at the top level, so a caller reads result.name', async () => {
        const [hit] = await store.findSimilar('pages', NORTH, 1)
        assert.equal(hit.name, 'north')
        assert.equal(hit.data, undefined, 'not nested under .data')
    })

    it('lets id and distance win over a same-named mapped field', async () => {
        // They are engine-provided facts. A map() that happens to emit `id`
        // must not be able to make a result point at the wrong entity.
        seedEntity('/real')
        await store.upsert('pages', '/real', NORTH, { id: '/lies', distance: 999, name: 'shadowed' })
        const hit = (await store.findSimilar('pages', NORTH, 5)).find(h => h.name === 'shadowed')
        assert.equal(hit.id, '/real')
        assert.notEqual(hit.distance, 999)
    })

    it('returns nothing from an empty store rather than throwing', async () => {
        assert.deepEqual(await store.findSimilar('media', NORTH, 5), [])
    })
})

describe('delete', () => {
    it('removes the row from BOTH tables, not just the ids table', async () => {
        // vec0 does not observe the delete on its own; a trigger propagates it.
        // If that trigger is wrong the entity keeps turning up in results while
        // the ids table says it is gone — a stale hit with no data.
        seedEntity('/documents/a.md')
        await store.upsert('pages', '/documents/a.md', NORTH, { title: 'A' })
        await store.delete('pages', '/documents/a.md')

        assert.equal(db.prepare('SELECT count(*) AS n FROM mikser_vector_pages_ids').get().n, 0)
        assert.equal(db.prepare('SELECT count(*) AS n FROM mikser_vector_pages').get().n, 0)
        assert.deepEqual(await store.findSimilar('pages', NORTH, 5), [])
    })

    it('is a no-op for an entity that was never stored', async () => {
        await assert.doesNotReject(() => store.delete('pages', '/never-here'))
    })
})

describe('catalog cascade', () => {
    it('purges vector rows when the ENTITY is deleted from the catalog', async () => {
        // The whole reason the ids table exists: vec0 cannot carry a foreign
        // key, so deletes route through a table that can, and a trigger carries
        // them the last step. Without this an entity deleted from the catalog
        // keeps answering searches forever.
        seedEntity('/documents/a.md')
        await store.upsert('pages', '/documents/a.md', NORTH, { title: 'A' })

        db.prepare('DELETE FROM mikser_entities WHERE id = ?').run('/documents/a.md')

        assert.equal(db.prepare('SELECT count(*) AS n FROM mikser_vector_pages_ids').get().n, 0,
            'the FK cascade should have emptied the ids table')
        assert.equal(db.prepare('SELECT count(*) AS n FROM mikser_vector_pages').get().n, 0,
            'and the trigger should have carried it into vec0')
        assert.deepEqual(await store.findSimilar('pages', NORTH, 5), [])
    })

    it('cascades across every store the entity appears in', async () => {
        seedEntity('/documents/a.md')
        await store.upsert('pages', '/documents/a.md', NORTH, { in: 'pages' })
        await store.upsert('media', '/documents/a.md', NORTH, { in: 'media' })

        db.prepare('DELETE FROM mikser_entities WHERE id = ?').run('/documents/a.md')

        assert.deepEqual(await store.findSimilar('pages', NORTH, 5), [])
        assert.deepEqual(await store.findSimilar('media', NORTH, 5), [])
    })

    it('leaves other entities alone', async () => {
        seedEntity('/keep'); seedEntity('/drop')
        await store.upsert('pages', '/keep', NORTH, { name: 'keep' })
        await store.upsert('pages', '/drop', EAST, { name: 'drop' })

        db.prepare('DELETE FROM mikser_entities WHERE id = ?').run('/drop')

        const hits = await store.findSimilar('pages', NORTH, 5)
        assert.deepEqual(hits.map(h => h.name), ['keep'])
    })
})

describe('clear', () => {
    it('empties every store', async () => {
        seedEntity('/a')
        await store.upsert('pages', '/a', NORTH, { n: 1 })
        await store.upsert('media', '/a', EAST, { n: 2 })
        await store.clear()
        assert.deepEqual(await store.findSimilar('pages', NORTH, 5), [])
        assert.deepEqual(await store.findSimilar('media', EAST, 5), [])
    })

    it('resets the rowid sequence, so a rebuild does not drift the ids upward', async () => {
        seedEntity('/a')
        await store.upsert('pages', '/a', NORTH, { n: 1 })
        await store.clear()
        await store.upsert('pages', '/a', NORTH, { n: 2 })
        assert.equal(db.prepare('SELECT rowid FROM mikser_vector_pages_ids').get().rowid, 1)
    })

    it('leaves the entity rows it does not own alone', async () => {
        seedEntity('/a')
        await store.upsert('pages', '/a', NORTH, { n: 1 })
        await store.clear()
        assert.equal(db.prepare('SELECT count(*) AS n FROM mikser_entities').get().n, 1)
    })
})

describe('close', () => {
    it('does not close the connection, which the engine owns', async () => {
        await store.close()
        assert.equal(db.isOpen, true)
    })
})
