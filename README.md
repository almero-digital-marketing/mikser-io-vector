# mikser-io-vector

OpenAI embeddings + [sqlite-vec](https://www.npmjs.com/package/sqlite-vec) or [pgvector](https://github.com/pgvector/pgvector) storage and search for [mikser-io](https://github.com/almero-digital-marketing/mikser-io). Indexes entities as they flow through the lifecycle, exposes a `findSimilar()` runtime helper, and (when a shared Express app is available) mounts a `POST /vector/:storeName` HTTP search endpoint.

## Why semantic search inside an SSG

At 10k+ documents, search isn't optional — users won't navigate that much content by directory listing or sitemap. Every large content project ends up with a search story; the question is where that story lives.

The usual answer is "integrate Algolia / Typesense / Meilisearch externally" — a separate index pipeline, a separate vendor bill, a separate auth model, and a "keep the index in sync with the content" problem that bites the moment a publish event fires.

This plugin builds the index inside mikser's normal lifecycle. When a document is created, updated, or deleted, the watcher fires, the entity gets re-embedded, and the search index updates — in the same process, on the same journal, alongside every other build step. One pipeline, no second vendor, no out-of-band drift.

The embedding source object is also stored alongside each vector and returned with search results — so the frontend can render the hit (title, summary, image) without a second lookup.

## Install

```bash
npm install mikser-io-vector
```

## Configure

```js
// mikser.config.js
export default {
  plugins: ['documents', 'layouts', 'render-hbs', 'api', 'vector'],

  vector: {
    // Backend selection. Defaults to local sqlite-vec.
    //   'better-sqlite3' | 'sqlite' | 'sqlite3'     → sqlite-vec
    //   'pg' | 'postgres' | 'postgresql'            → pgvector
    client: 'better-sqlite3',

    // Connection — interpreted per driver:
    //   sqlite: { filename } (defaults to <runtimeFolder>/vectors.db)
    //   pg:     a libpq URL string, or pg.PoolConfig, or omit and use
    //           PGHOST / PGUSER / PGPASSWORD / PGDATABASE / PGSSLMODE.
    // connection: process.env.DATABASE_URL,

    openai: {
      apiKey: process.env.OPENAI_API_KEY,    // or set OPENAI_API_KEY directly
      model: 'text-embedding-3-small',       // default
      dim: 1536,                              // default; must match the model
      // baseURL: 'https://...',              // optional, for Azure / self-hosted
    },

    base: '/vector',                      // HTTP mount path; default '/vector'
    concurrency: 4,                       // parallel OpenAI calls per store; default 4 — per-store override via stores[name].concurrency

    // Multiple named stores. Mirrors the data plugin's
    // (query, map, pick) shape so the same mental model applies.
    stores: {
      documents: {
        // Which entities go into this store. Defaults to
        // `entity => entity.type === 'document'` when omitted.
        // query: entity => entity.type === 'document',

        // Either return a plain object from `map`...
        map: async (entity) => ({
          title: entity.meta?.title,
          tags: entity.meta?.tags,
          content: entity.content,
        }),

        // ...OR a `pick` list of paths.
        // pick: ['meta.title', 'meta.tags', 'content'],
      },

      // Add as many stores as you need; each gets its own vec0 table.
      layouts: {
        query: entity => entity.type === 'layout',
        pick: ['name'],

        // Optional: protect this store's HTTP endpoint with a bearer token.
        // Programmatic findSimilar() is unaffected — auth is HTTP-only.
        token: process.env.VECTOR_LAYOUTS_TOKEN,
      },
    },
  },
}
```

Provide your OpenAI key either inline (`vector.openai.apiKey`) or as `OPENAI_API_KEY` in the environment.

## How it indexes

The plugin hooks `onBeforeRender` and iterates the journal for `CREATE`, `UPDATE`, and `DELETE` operations. For each store:

1. Apply `query(entity)` to filter — defaults to `entity => entity.type === 'document'` when not provided.
2. Build a plain object via `map(entity)` (async, must return an object) or `pick` (path → value). If both are empty, `entity.content` is embedded as-is.
3. Serialize the object via [TOON](https://github.com/toon-format/toon) — a compact, schema-aware textual format that's lighter on tokens than JSON and gives the embedding model a cleaner signal than ad-hoc string concatenation.
4. Compute the embedding via OpenAI and upsert into the store's vec0 table.
5. Deletes remove the vector and its rowid mapping.

In watch mode, only changed entities are re-embedded each cycle. In a one-shot build every CREATE re-embeds — keep that in mind for API cost.

## Search — programmatic

```js
import { runtime } from 'mikser-io'
// after runtime.start() once the plugin's onLoaded ran

const results = await runtime.findSimilar('documents', 'how do I publish a report', { limit: 5 })
// → [
//     {
//       id: '/documents/en/report.md',
//       distance: 0.123,
//       data: { title: 'Mikser Quarterly Report', content: '...' },
//     },
//     ...
//   ]
```

`data` is the *original object* returned by your `map(entity)` (or built from `pick`) — the thing that was TOON-encoded before embedding. Use it to surface human-readable metadata alongside the score without a second lookup.

## Search — HTTP

Requires a shared Express app (`--server` or `setup({ app })`). The plugin mounts `POST /vector/:storeName`:

```bash
curl -X POST http://localhost:3001/vector/documents \
  -H 'content-type: application/json' \
  -d '{ "q": "how do I publish a report", "limit": 5 }'

# {
#   "results": [
#     {
#       "id": "/documents/en/report.md",
#       "distance": 0.123,
#       "data": { "title": "Mikser Quarterly Report", "content": "..." }
#     },
#     ...
#   ]
# }
```

`q` is required; `limit` defaults to 5.

### Authentication

A store may declare a `token` — when set, its HTTP endpoint requires `Authorization: Bearer <token>`. Stores without a `token` remain open. The programmatic `runtime.findSimilar()` is never gated by tokens.

```bash
curl -X POST http://localhost:3001/vector/layouts \
  -H 'authorization: Bearer s3cr3t' \
  -H 'content-type: application/json' \
  -d '{ "q": "report layout", "limit": 3 }'

# Missing/wrong token → 401 { "error": "Invalid or missing token" }
```

## Storage

**sqlite-vec (`client: 'better-sqlite3'`)** — vectors live in `<runtimeFolder>/vectors.db`. Each configured store has two tables: `mikser_vector_<storeName>` (the vec0 virtual table) and `mikser_vector_<storeName>_ids` (a regular table mapping string `entity_id` to numeric `rowid` and holding the JSON `data` payload). Wipe with `--clear` to start fresh — every entity will be re-embedded on the next run.

**pgvector (`client: 'pg'`)** — one table per store: `mikser_vector_<storeName> (id TEXT PRIMARY KEY, embedding vector(N), data jsonb)`, plus an HNSW index using `vector_cosine_ops`. Requires the `vector` extension on the database (Neon and Supabase have it pre-installed; vanilla Postgres needs `CREATE EXTENSION vector` by a superuser). `--clear` `TRUNCATE`s every configured store table so the next run re-embeds from scratch.

Both backends use **cosine distance**, so values are comparable when switching backends with the same embedding model.

## Notes

- sqlite-vec uses FLAT (brute-force) search — plenty fast up to ~100K vectors. Beyond that, use pgvector with its HNSW index.
- Embedding model and dimensions can be changed, but the existing schema is fixed at create time. If you change `dim`, drop the vector tables so they get re-created.
- The plugin requires `runtime.options.app` for HTTP search but not for programmatic search — `findSimilar()` works either way.

## License

MIT
