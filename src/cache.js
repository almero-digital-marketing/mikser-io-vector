// The embedding cache: the one thing here that money bought.
//
// Everything else this plugin stores is derived in the ordinary sense — the
// vec0 index, the ids table, the mapped `data` — and all of it lives in
// `mikser.sqlite`, which the engine UNLINKS whenever the cache is invalid:
// `--clear`, a config edit, a schema fingerprint that moved on upgrade (see
// ADR-0002, files are the source of truth). Rebuilding it costs sqlite
// inserts, which is as it should be.
//
// An embedding is not like that. It is derived from a file too, but it is
// derived by a paid API call over a network, and a site with 11,000 documents
// pays 11,000 of them to get back exactly the vectors it already had. Editing
// one line of config was enough to trigger it, which is the same shape of
// defect `mikser-io-assets` had: the expensive derived thing destroyed as a
// side effect of invalidating the cheap one.
//
// So the vectors are kept HERE, in a database of this plugin's own, keyed on
// the text that produced them and on the model that produced them. The engine
// never touches this file: `--clear` removes everything in the runtime folder
// EXCEPT sqlite artefacts (see the engine's boot.js), which is why the name
// must keep its `.sqlite` extension, and the cache wipe next door unlinks one
// named file rather than a directory.
//
// It is append-only. Nothing in here expires, because evicting an embedding
// means paying for it again, and a cache that quietly decides to do that is
// the thing this file exists to prevent. `--clear-vectors` empties it when
// that is what you actually want. At 1536 dimensions a row is about 6KB, so
// 11,000 documents is roughly 66MB — the file is yours to delete.

import path from 'node:path'
import { createHash } from 'node:crypto'
import Database from 'better-sqlite3'

export const DEFAULT_CACHE_FILENAME = 'mikser.vector.sqlite'

// provider + model + dim + text. The model is in the key because two models
// answer the same text with incomparable vectors, and dim is in it because a
// provider that changes a model's width underneath you would otherwise hand
// back a blob the vec0 table cannot take.
export function embeddingKey({ provider, model, dim, text }) {
    return createHash('sha1')
        .update(`${provider}\u0000${model}\u0000${dim}\u0000${text}`)
        .digest('hex')
}

export function createEmbeddingCache({ filename, logger }) {
    const database = new Database(filename)
    // WAL so a reader (another mikser process asking the same question) is
    // not blocked by the writer, and NORMAL because losing the last few
    // embeddings to a power cut costs a few API calls, not correctness.
    database.pragma('journal_mode = WAL')
    database.pragma('synchronous = NORMAL')
    database.exec(`
        CREATE TABLE IF NOT EXISTS mikser_vector_embeddings (
            key TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at INTEGER NOT NULL
        )
    `)

    const selectOne = database.prepare(
        'SELECT embedding, dim FROM mikser_vector_embeddings WHERE key = ?')
    const insertOne = database.prepare(`
        INSERT INTO mikser_vector_embeddings (key, provider, model, dim, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET embedding = excluded.embedding
    `)
    const countAll = database.prepare('SELECT COUNT(*) AS count FROM mikser_vector_embeddings')

    return {
        path: filename,

        // The stored vector, or null for a miss.
        //
        // A row whose blob is not `dim` floats wide is a MISS rather than an
        // error: it can only come from a provider that changed a model's
        // width without changing its name, and the honest answer to that is
        // to embed again and overwrite, not to fail a build.
        get(key, dim) {
            const row = selectOne.get(key)
            if (!row) return null
            if (row.dim !== dim || row.embedding.byteLength !== dim * 4) {
                logger?.debug('Vector cache: stored width %d does not match %d, re-embedding', row.dim, dim)
                return null
            }
            // Copied out of the buffer better-sqlite3 hands back. It is a
            // view into a pooled allocation, and a Float32Array built over it
            // directly would read whatever the pool holds next.
            const bytes = row.embedding
            return new Float32Array(
                bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
        },

        set(key, vector, { provider, model, dim }) {
            insertOne.run(
                key, provider, model, dim,
                Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength),
                Date.now())
        },

        count() {
            return countAll.get().count
        },

        // Returns how many went, because the log line is the only evidence
        // that an irreversible thing happened on purpose.
        clear() {
            const before = countAll.get().count
            database.exec('DELETE FROM mikser_vector_embeddings')
            database.exec('VACUUM')
            return before
        },

        close() {
            database.close()
        },
    }
}

// Where the cache file goes, and a refusal rather than a silent loss.
//
// Inside the runtime folder the extension is load-bearing: `--clear` deletes
// every entry there except `.sqlite`, `.sqlite-wal` and `.sqlite-shm`, so a
// cache named `vectors.db` would be thrown away by the one command this whole
// file exists to survive. Outside it, nothing deletes anything and the name
// is the caller's business.
export function resolveCachePath({ cache, runtimeFolder }) {
    const filename = typeof cache === 'string' && cache.length ? cache : DEFAULT_CACHE_FILENAME
    const resolved = path.isAbsolute(filename) ? filename : path.join(runtimeFolder, filename)
    const inRuntimeFolder = !path.relative(runtimeFolder, resolved).startsWith('..')
    if (inRuntimeFolder && !resolved.endsWith('.sqlite')) {
        throw new Error(
            `vector({ cache: ${JSON.stringify(filename)} }): a cache inside the runtime folder must end `
            + 'in .sqlite, or `mikser --clear` deletes it — and it is the only thing here that '
            + 'a rebuild cannot reproduce without paying the model again.')
    }
    return resolved
}
