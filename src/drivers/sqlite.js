// sqlite-vec driver: stores vectors in a vec0 virtual table per store
// plus a companion `<store>_ids` table mapping string entity_id → integer
// rowid and holding the JSON `data` payload returned with search results.
import path from 'node:path'
import Database from 'better-sqlite3'
import * as sqliteVec from 'sqlite-vec'

const vecTable = (storeName) => `vec_${storeName}`
const idsTable = (storeName) => `vec_${storeName}_ids`

export async function createDriver({ runtime, dim, stores, connection }) {
    const dbPath = connection?.filename
        ?? path.join(runtime.options.runtimeFolder, 'vectors.db')
    const db = new Database(dbPath)
    sqliteVec.load(db)

    for (const storeName of Object.keys(stores)) {
        // distance_metric=cosine keeps L2-on-normalized and cosine
        // distances comparable across backends. OpenAI embeddings are
        // unit-normalized so cosine is the natural metric.
        db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS ${vecTable(storeName)} USING vec0(
            embedding float[${dim}] distance_metric=cosine,
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

    return {
        describe: () => `sqlite ${dbPath}`,

        async upsert(storeName, entityId, vec, data) {
            const buf = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength)
            const dataJson = data == null ? null : JSON.stringify(data)
            const existing = db.prepare(
                `SELECT rowid FROM ${idsTable(storeName)} WHERE entity_id = ?`
            ).get(entityId)
            if (existing) {
                db.prepare(`UPDATE ${vecTable(storeName)} SET embedding = ? WHERE rowid = ?`)
                    .run(buf, existing.rowid)
                db.prepare(`UPDATE ${idsTable(storeName)} SET data = ? WHERE rowid = ?`)
                    .run(dataJson, existing.rowid)
            } else {
                const res = db.prepare(
                    `INSERT INTO ${idsTable(storeName)} (entity_id, data) VALUES (?, ?)`
                ).run(entityId, dataJson)
                db.prepare(
                    `INSERT INTO ${vecTable(storeName)}(rowid, embedding, entity_id) VALUES (?, ?, ?)`
                ).run(BigInt(res.lastInsertRowid), buf, entityId)
            }
        },

        async delete(storeName, entityId) {
            const existing = db.prepare(
                `SELECT rowid FROM ${idsTable(storeName)} WHERE entity_id = ?`
            ).get(entityId)
            if (!existing) return
            db.prepare(`DELETE FROM ${vecTable(storeName)} WHERE rowid = ?`).run(existing.rowid)
            db.prepare(`DELETE FROM ${idsTable(storeName)} WHERE rowid = ?`).run(existing.rowid)
        },

        async findSimilar(storeName, vec, limit) {
            const buf = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength)
            // vec0 requires LIMIT (or k = ?) directly on the MATCH query,
            // so we KNN-match in a subquery and join the ids/data table
            // outside.
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
        },

        async close() { db.close() },
    }
}
