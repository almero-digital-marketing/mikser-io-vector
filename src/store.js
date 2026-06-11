// Sqlite-vec backed vector store, layered onto the engine's shared
// `mikser.sqlite` connection. The plugin loads sqlite-vec on that
// connection once at setup, then registers per-store vec0 virtual
// tables (`mikser_vector_<store>`) plus companion ids tables
// (`mikser_vector_<store>_ids`) via `registerSchema` — applied
// immediately because the engine database is already open by the time
// the vector plugin's `onLoaded` runs.
//
// One substrate, one file (`runtime/mikser.sqlite`), one set of WAL /
// foreign_keys / synchronous pragmas. Deleting an entity from the
// catalog cascades to its vector rows via the FK on
// `mikser_vector_<store>_ids.entity_id → mikser_entities.id`.
//
// Sqlite + sqlite-vec is the only backend. The pluggable
// sqlite-or-postgres abstraction was dropped during the engine's
// catalog migration (postgres-async vs template-sync wall); vector
// follows the same shape.
const vecTable = (storeName) => `mikser_vector_${storeName}`
const idsTable = (storeName) => `mikser_vector_${storeName}_ids`

export async function createStore({ db, dim, stores, registerSchema }) {
    if (!db?.isOpen) {
        throw new Error('createStore requires an open engine database handle (useDatabase()).')
    }

    // sqlite-vec is loaded into the engine connection via
    // loadExtension() at the plugin's module-eval (see index.js). The
    // engine substrate runs all loadExtension callbacks inside its
    // setupConnection, after the raw new Database() but before any
    // schema apply — so vec0 is available before any vec0 CREATE
    // here, and before any other onLoaded prepares a statement that
    // would otherwise trip the schema validator on second-and-later
    // opens.

    for (const storeName of Object.keys(stores)) {
        // distance_metric=cosine: OpenAI embeddings are unit-normalized,
        // so cosine is the natural metric and stays comparable to
        // L2-on-normalized.
        registerSchema(vecTable(storeName), `
            CREATE VIRTUAL TABLE IF NOT EXISTS ${vecTable(storeName)} USING vec0(
                embedding float[${dim}] distance_metric=cosine,
                +entity_id TEXT
            )
        `)
        // FK to mikser_entities.id with ON DELETE CASCADE so catalog
        // deletes auto-purge orphaned vector rows. The companion vec0
        // row is dropped via the trigger below (vec0 virtual tables
        // don't support FK directly, so we route deletes through the
        // ids table).
        registerSchema(idsTable(storeName), `
            CREATE TABLE IF NOT EXISTS ${idsTable(storeName)} (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL UNIQUE
                    REFERENCES mikser_entities(id) ON DELETE CASCADE,
                data TEXT
            )
        `)
        // When the catalog deletes an entity, the FK cascades into the
        // ids table; this trigger then propagates the delete into the
        // vec0 virtual table by rowid (vec0 doesn't observe FK cascades
        // directly).
        registerSchema(`${idsTable(storeName)}_cascade_trigger`, `
            CREATE TRIGGER IF NOT EXISTS ${idsTable(storeName)}_cascade
            AFTER DELETE ON ${idsTable(storeName)}
            BEGIN
                DELETE FROM ${vecTable(storeName)} WHERE rowid = OLD.rowid;
            END
        `)
    }

    return {
        describe: () => `sqlite-vec on engine db (${db.path})`,

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
            // The trigger handles the vec0 side; we only need to drop
            // the ids row. (When the deletion arrives via FK cascade —
            // catalog deleted the entity — the same trigger fires.)
            db.prepare(`DELETE FROM ${idsTable(storeName)} WHERE entity_id = ?`).run(entityId)
        },

        async findSimilar(storeName, vec, limit) {
            const buf = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength)
            // vec0 requires LIMIT (or k = ?) directly on the MATCH query,
            // so KNN-match in a subquery and join the ids/data table
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

        async clear() {
            for (const storeName of Object.keys(stores)) {
                db.exec(`DELETE FROM ${vecTable(storeName)}`)
                db.exec(`DELETE FROM ${idsTable(storeName)}`)
                db.exec(`DELETE FROM sqlite_sequence WHERE name = '${idsTable(storeName)}'`)
            }
        },

        // No-op: the engine owns the connection and closes it.
        async close() {},
    }
}
