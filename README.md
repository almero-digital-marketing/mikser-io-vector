# mikser-io-vector

Provider-agnostic embeddings (via the [Vercel AI SDK](https://sdk.vercel.ai/)) + [sqlite-vec](https://www.npmjs.com/package/sqlite-vec) storage and search for [mikser-io](https://github.com/almero-digital-marketing/mikser-io). Indexes entities as they flow through the lifecycle, exposes a `findSimilar()` runtime helper, and (when a shared Express app is available) mounts a `POST /vector/:storeName` HTTP search endpoint.

## Why semantic search inside an SSG

At 10k+ documents, search isn't optional — users won't navigate that much content by directory listing or sitemap. Every large content project ends up with a search story; the question is where that story lives.

The usual answer is "integrate Algolia / Typesense / Meilisearch externally" — a separate index pipeline, a separate vendor bill, a separate auth model, and a "keep the index in sync with the content" problem that bites the moment a publish event fires.

This plugin builds the index inside mikser's normal lifecycle. When a document is created, updated, or deleted, the watcher fires, the entity gets re-embedded, and the search index updates — in the same process, on the same journal, alongside every other build step. One pipeline, no second vendor, no out-of-band drift.

The embedding source object is also stored alongside each vector and returned with search results — so the frontend can render the hit (title, summary, image) without a second lookup.

## Install

```bash
npm install mikser-io-vector
# plus the AI-SDK provider for whichever embedding model you want:
npm install @ai-sdk/openai           # or @ai-sdk/anthropic, @ai-sdk/mistral, @ai-sdk/cohere, @ai-sdk/google, etc.
```

The plugin is provider-agnostic — it talks to whatever embedding model you hand it, as long as it implements the AI SDK's `EmbeddingModelV2` spec. Switch providers by changing one line in your config.

## Configure

```js
// mikser.config.js
import { documents, files, renderHbs, api } from 'mikser-io'
import { layouts } from 'mikser-io-layouts'
import { vector } from 'mikser-io-vector'
import { openai } from '@ai-sdk/openai'   // or @ai-sdk/anthropic, etc.

export default {
  plugins: [
    documents(),
    layouts(),
    renderHbs(),
    api(),
    vector({
      // The embedding model is a Vercel AI SDK model factory return.
      // Provider, model name, and any provider-specific options live
      // here — vector doesn't reinterpret any of that.
      model: openai.embedding('text-embedding-3-small'),
      // dim is probed automatically from a one-shot sample embedding
      // at startup, so it always matches the model's actual output.

      base: '/vector',                      // HTTP mount path; default '/vector'
      concurrency: 4,                       // parallel embedding calls per store; default 4 — per-store override via stores[name].concurrency
      cache: 'mikser.vector.sqlite',        // where the embeddings are kept, relative to the runtime folder.
                                            // See "The embedding cache"; a name inside the runtime folder
                                            // must end in .sqlite, or --clear would delete it

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
    }),
  ],
}
```

Provider credentials live where the AI SDK provider expects them — typically `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / etc. in the environment. The vector plugin holds no auth state of its own; it just hands the model to the SDK.

### Switching provider

```js
// OpenAI (default in most examples)
import { openai } from '@ai-sdk/openai'
vector({ model: openai.embedding('text-embedding-3-small'), stores })

// Anthropic
import { anthropic } from '@ai-sdk/anthropic'
vector({ model: anthropic.embedding('voyage-3'), stores })

// Mistral
import { mistral } from '@ai-sdk/mistral'
vector({ model: mistral.textEmbeddingModel('mistral-embed'), stores })

// Local via Ollama (through the openai-compatible provider)
import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
const ollama = createOpenAICompatible({ baseURL: 'http://localhost:11434/v1', name: 'ollama' })
vector({ model: ollama.embedding('nomic-embed-text'), stores })
```

Changing model also changes `dim` (probed at startup), and the index is built at that width. Editing the config is enough to rebuild it: mikser wipes its cache on a config change, and the next run re-creates the tables at the new width. The embeddings are re-computed too, because the cache is keyed on the model — see [The embedding cache](#the-embedding-cache).

## How it indexes

The plugin hooks `onBeforeRender` and iterates the journal for `CREATE`, `UPDATE`, and `DELETE` operations. For each store:

1. Apply `query(entity)` to filter — defaults to `entity => entity.type === 'document'` when not provided.
2. Build a plain object via `map(entity)` (async, must return an object) or `pick` (path → value). If both are empty, `entity.content` is embedded as-is.
3. Serialize the object via [TOON](https://github.com/toon-format/toon) — a compact, schema-aware textual format that's lighter on tokens than JSON and gives the embedding model a cleaner signal than ad-hoc string concatenation.
4. Look the (model, text) pair up in the embedding cache; compute it via the configured AI-SDK model only on a miss; upsert into the store's vec0 table either way.
5. Deletes remove the vector and its rowid mapping.

In watch mode, only changed entities are indexed each cycle. A cold build indexes everything, but indexing is not the same as embedding: step 4 asks the cache first, and only an unseen (model, text) pair reaches the provider.

## The embedding cache

The vec0 index, the ids table and the mapped `data` all live in mikser's own
`mikser.sqlite`, which the engine unlinks whenever the cache is invalid —
`--clear`, a config edit, a schema fingerprint that moved on upgrade. That is
correct for derived state: files are the source of truth and rebuilding costs
sqlite inserts.

An embedding is derived too, but it is derived by a paid API call. Editing one
line of config used to buy one call per indexed entity to get back exactly the
vectors the site already had — on a site with 11,000 documents, 11,000 of them.

So the vectors themselves are kept in a database of this plugin's own,
`<runtimeFolder>/mikser.vector.sqlite`, keyed on the text that produced them
and the model that produced it. Nothing the engine does touches it: `--clear`
empties the runtime folder except for sqlite artefacts, which is why the name
has to keep its `.sqlite` extension — pass `cache` to move it, and a name
inside the runtime folder that `--clear` would delete is refused rather than
silently lost.

A rebuilt index therefore costs inserts, not calls:

```
Vector: 0 embedded, 11003 reused from the cache
```

Changing the model, or changing what `map()`/`pick` feeds it, changes the key
and re-embeds — no flag needed, because the text really is different.

The cache is append-only. Nothing in it expires, because evicting an embedding
means paying for it again, and a cache that quietly decides to do that is the
thing this exists to prevent. At 1536 dimensions a row is about 6KB, so 11,000
documents is roughly 66MB; the file is yours to delete.

`--clear-vectors` empties it. On its own that only stops later cycles reusing
what is there — a cycle re-embeds the entities it is given, and on a build
where nothing changed that is none of them. Pair it with `--clear` to re-embed
the whole site. The index is deliberately left alone either way: emptying it
would take search down until something re-indexed.

## Search — programmatic

```js
import { runtime } from 'mikser-io'
// after runtime.start() once the plugin's onLoaded ran

const results = await runtime.findSimilar('documents', 'how do I publish a report', { limit: 5 })
// → [
//     {
//       title: 'Mikser Quarterly Report',
//       content: '...',
//       id: '/documents/en/report.md',
//       distance: 0.123,
//     },
//     ...
//   ]
```

Each hit spreads the original object returned by your `map(entity)` (or built from `pick`) at the top level, plus `id` and `distance`. `map`'s output is the same thing that was TOON-encoded before embedding — use it to surface human-readable metadata alongside the score without a second lookup. `id` and `distance` are engine-set and override any same-named field from `map()`.

## Search — HTTP

Requires a shared Express app (`--server` or `setup({ app })`). The plugin mounts `POST /vector/:storeName`:

```bash
curl -X POST http://localhost:3001/vector/documents \
  -H 'content-type: application/json' \
  -d '{ "q": "how do I publish a report", "limit": 5 }'

# {
#   "results": [
#     {
#       "title": "Mikser Quarterly Report",
#       "content": "...",
#       "id": "/documents/en/report.md",
#       "distance": 0.123
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

The index lives in mikser's own `<runtimeFolder>/mikser.sqlite` (sqlite-vec loaded onto the engine's connection). Each configured store has two tables: `mikser_vector_<storeName>` (the vec0 virtual table) and `mikser_vector_<storeName>_ids` (a regular table mapping string `entity_id` to numeric `rowid` and holding the JSON `data` payload). A foreign key from the ids table into `mikser_entities` cascades a catalog delete into both.

The embeddings live beside it in `<runtimeFolder>/mikser.vector.sqlite`, which the engine never touches. `--clear` therefore rebuilds the index without re-embedding anything.

Distances are **cosine** — OpenAI embeddings are unit-normalized so cosine is the natural metric.

## Notes

- sqlite-vec uses FLAT (brute-force) search — plenty fast up to ~100K vectors.
- Embedding model and dimensions can be changed: the vec0 schema is fixed at create time, but it is created in mikser's cache, and editing the config wipes that — so a model swap rebuilds the tables at the new width on the next run.
- The plugin requires `runtime.options.app` for HTTP search but not for programmatic search — `findSimilar()` works either way.

## License

MIT
