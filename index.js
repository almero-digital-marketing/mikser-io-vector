import path from 'node:path'
import Database from 'better-sqlite3'
import * as sqliteVec from 'sqlite-vec'
import OpenAI from 'openai'
import _ from 'lodash'
import { encode as toonEncode } from '@toon-format/toon'

export default ({
    runtime,
    onLoaded,
    onBeforeRender,
    useLogger,
    useJournal,
    constants: { OPERATION },
}) => {
    const config = runtime.config.vector ?? {}
    const stores = config.stores ?? {}

    let db
    let openai
    let model
    let dim

    // sqlite-vec stores embeddings keyed by INTEGER rowid; the rest of
    // mikser's world keys on string entity ids. Each store gets a
    // companion `<vecTable>_ids` table that maps entity_id → rowid so
    // updates and deletes can find the right row without scanning the
    // vec0 table on its (unindexed) auxiliary `+entity_id` column.
    const vecTable = (storeName) => `vec_${storeName}`
    const idsTable = (storeName) => `vec_${storeName}_ids`

    async function embed(text) {
        const { data } = await openai.embeddings.create({
            model,
            input: text,
            dimensions: dim,
        })
        return new Float32Array(data[0].embedding)
    }

    function upsertVector(storeName, entityId, vec, data) {
        const buf = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength)
        const dataJson = data == null ? null : JSON.stringify(data)
        const existing = db.prepare(
            `SELECT rowid FROM ${idsTable(storeName)} WHERE entity_id = ?`
        ).get(entityId)
        if (existing) {
            db.prepare(
                `UPDATE ${vecTable(storeName)} SET embedding = ? WHERE rowid = ?`
            ).run(buf, existing.rowid)
            db.prepare(
                `UPDATE ${idsTable(storeName)} SET data = ? WHERE rowid = ?`
            ).run(dataJson, existing.rowid)
        } else {
            const res = db.prepare(
                `INSERT INTO ${idsTable(storeName)} (entity_id, data) VALUES (?, ?)`
            ).run(entityId, dataJson)
            db.prepare(
                `INSERT INTO ${vecTable(storeName)}(rowid, embedding, entity_id) VALUES (?, ?, ?)`
            ).run(BigInt(res.lastInsertRowid), buf, entityId)
        }
    }

    function deleteVector(storeName, entityId) {
        const existing = db.prepare(
            `SELECT rowid FROM ${idsTable(storeName)} WHERE entity_id = ?`
        ).get(entityId)
        if (!existing) return
        db.prepare(`DELETE FROM ${vecTable(storeName)} WHERE rowid = ?`).run(existing.rowid)
        db.prepare(`DELETE FROM ${idsTable(storeName)} WHERE rowid = ?`).run(existing.rowid)
    }

    async function findSimilar(storeName, text, { limit = 5 } = {}) {
        if (!stores[storeName]) {
            throw new Error(`Unknown vector store: ${storeName}`)
        }
        const vec = await embed(text)
        const buf = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength)
        // vec0 requires LIMIT (or k = ?) directly on the MATCH query, so
        // we KNN-match in a subquery and join the ids/data table outside.
        const rows = db.prepare(`
            SELECT v.entity_id AS id, v.distance AS distance, i.data AS data
            FROM (
                SELECT rowid, entity_id, distance
                FROM ${vecTable(storeName)}
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            ) v
            JOIN ${idsTable(storeName)} i ON i.rowid = v.rowid
            ORDER BY v.distance
        `).all(buf, limit)
        return rows.map(r => ({
            id: r.id,
            distance: r.distance,
            data: r.data ? JSON.parse(r.data) : null,
        }))
    }

    // Build what we need from an entity: `data` is the original mapped
    // object (returned to callers of findSimilar), `text` is the
    // TOON-encoded form fed to the embedding model. Mirrors the data
    // plugin's (query, map, pick) shape. Returns null when there's
    // nothing to embed.
    async function entityToEmbedding(entity, { map, pick }) {
        let data
        if (map) {
            const obj = await map(entity)
            if (obj == null) return null
            if (typeof obj !== 'object' || Array.isArray(obj)) {
                throw new Error(
                    `Vector store map() must return a plain object; got ${Array.isArray(obj) ? 'array' : typeof obj}`
                )
            }
            data = obj
        } else if (pick) {
            data = {}
            for (const p of pick) {
                const v = _.get(entity, p)
                if (v !== undefined && v !== null && v !== '') data[p] = v
            }
        } else if (entity.content) {
            data = { content: entity.content }
        } else {
            return null
        }
        if (!Object.keys(data).length) return null
        return { data, text: toonEncode(data) }
    }

    onLoaded(async () => {
        const logger = useLogger()

        const apiKey = config.openai?.apiKey ?? process.env.OPENAI_API_KEY
        if (!apiKey) {
            throw new Error(
                'Vector plugin requires an OpenAI API key. ' +
                'Set vector.openai.apiKey in mikser.config.js, ' +
                'or export OPENAI_API_KEY in the environment.'
            )
        }
        openai = new OpenAI({
            apiKey,
            baseURL: config.openai?.baseURL,
        })
        model = config.openai?.model ?? 'text-embedding-3-small'
        dim = config.openai?.dim ?? 1536

        const dbPath = path.join(runtime.options.runtimeFolder, 'vectors.db')
        db = new Database(dbPath)
        sqliteVec.load(db)

        for (const storeName of Object.keys(stores)) {
            db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS ${vecTable(storeName)} USING vec0(
                embedding float[${dim}],
                +entity_id TEXT
            )`)
            db.exec(`CREATE TABLE IF NOT EXISTS ${idsTable(storeName)} (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL UNIQUE,
                data TEXT
            )`)
            // Pre-1.1.0 databases lacked the data column; add it if missing
            // so existing vectors.db files keep working.
            try { db.exec(`ALTER TABLE ${idsTable(storeName)} ADD COLUMN data TEXT`) } catch {}
        }

        logger.info(
            'Vector store initialized: %s (model=%s, dim=%d, stores=[%s])',
            dbPath, model, dim, Object.keys(stores).join(', ') || '<none>'
        )

        // Expose programmatic API
        runtime.findSimilar = findSimilar

        // HTTP endpoint when a shared Express app is around (--server or
        // setup({app})). Falls back to no-op otherwise; the programmatic
        // findSimilar() still works.
        if (runtime.options.app) {
            const { default: express } = await import('express').catch(() => ({ default: null }))
            if (!express) {
                logger.warn('Vector: express not available, HTTP search endpoint disabled')
                return
            }
            const base = config.base ?? '/vector'
            const router = express.Router()
            router.use(express.json())

            // POST /api/vector/:storeName  body: { q, limit }
            // If the store has a `token` configured, the request must carry
            // a matching `Authorization: Bearer <token>` header.
            router.post('/:storeName', async (req, res) => {
                try {
                    const { storeName } = req.params
                    const store = stores[storeName]
                    if (!store) {
                        return res.status(404).json({ error: `Unknown store: ${storeName}` })
                    }
                    if (store.token) {
                        const auth = req.get('authorization') ?? ''
                        const supplied = auth.startsWith('Bearer ') ? auth.slice(7) : null
                        if (supplied !== store.token) {
                            return res.status(401).json({ error: 'Invalid or missing token' })
                        }
                    }
                    const { q, limit = 5 } = req.body ?? {}
                    if (!q || typeof q !== 'string') {
                        return res.status(400).json({ error: 'q (string) is required in the request body' })
                    }
                    const results = await findSimilar(storeName, q, { limit })
                    res.json({ results })
                } catch (err) {
                    logger.error('Vector search error: %s', err.message)
                    res.status(500).json({ error: err.message })
                }
            })

            runtime.options.app.use(base, router)
            logger.info('Vector search mounted: %s/:storeName', base)
        }
    })

    onBeforeRender(async (signal) => {
        const logger = useLogger()
        if (!db || Object.keys(stores).length === 0) return

        for (const storeName of Object.keys(stores)) {
            const { query, map, pick } = stores[storeName]
            if (typeof query !== 'function') {
                logger.warn('Vector store %s has no query function; skipping', storeName)
                continue
            }

            for await (let { operation, entity } of useJournal(
                `Vector ${storeName}`,
                [OPERATION.CREATE, OPERATION.UPDATE, OPERATION.DELETE],
                signal,
            )) {
                if (!query(entity)) continue

                if (operation === OPERATION.DELETE) {
                    deleteVector(storeName, entity.id)
                    logger.trace('Vector deleted %s: %s', storeName, entity.id)
                    continue
                }

                const result = await entityToEmbedding(entity, { map, pick })
                if (!result) {
                    logger.trace('Vector skip %s — no text: %s', storeName, entity.id)
                    continue
                }

                try {
                    const vec = await embed(result.text)
                    upsertVector(storeName, entity.id, vec, result.data)
                    logger.debug('Vector embedded %s: %s', storeName, entity.id)
                } catch (err) {
                    logger.error('Vector embed error %s %s: %s', storeName, entity.id, err.message)
                }
            }
        }
    })
}
