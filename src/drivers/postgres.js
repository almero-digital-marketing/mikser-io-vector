// pgvector driver: one table per store with a vector(N) column, jsonb
// data, and an HNSW index for fast cosine search. Requires the pgvector
// extension to be available on the database (Neon, Supabase, RDS+ext,
// or a self-hosted postgres with pgvector installed).
import pg from 'pg'

const tableName = (storeName) => `vec_${storeName}`

export async function createDriver({ runtime, dim, stores, connection, logger }) {
    // `connection` is optional — when omitted, pg reads libpq env vars
    // (PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGSSLMODE, ...). String =
    // a connection URL; object = pg.PoolConfig.
    const poolConfig =
        typeof connection === 'string' ? { connectionString: connection }
        : connection ? connection
        : {}
    const pool = new pg.Pool({ max: 10, ...poolConfig })

    // Make sure the extension is available. Neon/Supabase have it
    // pre-installed; vanilla self-hosted postgres needs the pgvector
    // package and a superuser to create the extension.
    await pool.query('CREATE EXTENSION IF NOT EXISTS vector')

    for (const storeName of Object.keys(stores)) {
        await pool.query(`
            CREATE TABLE IF NOT EXISTS ${tableName(storeName)} (
                id TEXT PRIMARY KEY,
                embedding vector(${dim}),
                data jsonb
            )
        `)
        // HNSW + cosine ops. Index creation is idempotent and cheap
        // when the table is empty; on already-populated tables this is
        // a no-op when the index already exists.
        await pool.query(`
            CREATE INDEX IF NOT EXISTS ${tableName(storeName)}_embedding_idx
            ON ${tableName(storeName)}
            USING hnsw (embedding vector_cosine_ops)
        `)
    }

    // pgvector accepts the textual form '[v1,v2,...]' for input.
    const vecToString = (vec) => '[' + Array.from(vec).join(',') + ']'

    const host = (() => {
        if (typeof connection === 'string') {
            try { return new URL(connection).host } catch { return '<postgres>' }
        }
        if (connection?.connectionString) {
            try { return new URL(connection.connectionString).host } catch { return '<postgres>' }
        }
        return connection?.host ?? process.env.PGHOST ?? '<postgres>'
    })()

    return {
        describe: () => `postgres ${host}`,

        async upsert(storeName, entityId, vec, data) {
            await pool.query(
                `INSERT INTO ${tableName(storeName)} (id, embedding, data)
                 VALUES ($1, $2::vector, $3::jsonb)
                 ON CONFLICT (id) DO UPDATE
                 SET embedding = EXCLUDED.embedding, data = EXCLUDED.data`,
                [entityId, vecToString(vec), data == null ? null : JSON.stringify(data)]
            )
        },

        async delete(storeName, entityId) {
            await pool.query(
                `DELETE FROM ${tableName(storeName)} WHERE id = $1`,
                [entityId]
            )
        },

        async findSimilar(storeName, vec, limit) {
            const { rows } = await pool.query(
                `SELECT id, embedding <=> $1::vector AS distance, data
                 FROM ${tableName(storeName)}
                 ORDER BY distance
                 LIMIT $2`,
                [vecToString(vec), limit]
            )
            // node-pg returns jsonb as parsed JS objects already and
            // <=> produces a numeric (string in node-pg's text protocol);
            // coerce to Number for a clean shape.
            return rows.map(r => ({
                id: r.id,
                distance: Number(r.distance),
                data: r.data ?? null,
            }))
        },

        async close() { await pool.end() },
    }
}
