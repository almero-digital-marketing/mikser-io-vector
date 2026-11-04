import _ from 'lodash'
import pMap from 'p-map'
import * as sqliteVec from 'sqlite-vec'
import { encode as toonEncode } from '@toon-format/toon'
import { z } from 'zod'
import { embed as aiEmbed } from 'ai'
import { cliOption } from 'mikser-io'
import { createStore } from './src/store.js'
import { createEmbeddingCache, embeddingKey, resolveCachePath } from './src/cache.js'

export function vector(options = {}) {
    return ({
        runtime,
        onLoaded,
        onBeforeRender,
        useLogger,
        useJournal,
        useDatabase,
        registerSchema,
        registerRoute,
        registerTool,
        onProvision,
        constants: { OPERATION },
    }) => {
    onProvision(ctx => sqliteVec.load(ctx.handle))

    // Declared in the load phase, like every plugin option — the table is
    // sealed before onLoaded runs.
    cliOption('--clear-vectors',
        'empty the embedding cache. On its own it only stops future cycles reusing it; '
        + 'with --clear it re-embeds the whole site, at one paid API call per entity', false)

    const config = options
    const stores = config.stores ?? {}

    let store
    let model      // Vercel AI SDK embedding model — opaque object the user passed.
    let dim        // Probed once at init from a sample embedding; sqlite-vec
                   // needs it at schema-creation time.
    let cache      // The embeddings this site has already paid for. See src/cache.js.

    // Per-cycle, so the log line can say what the cache was worth.
    let embedded = 0
    let reused = 0

    // `remember` is what separates INDEXING from SEARCHING.
    //
    // An indexed entity's text comes from the working folder: a bounded set,
    // written again identically after every wipe, and therefore worth keeping
    // forever. A query comes from whoever is searching — an HTTP caller, an
    // agent asking the MCP tool twelve different ways — and storing those
    // would make an append-only cache grow with traffic rather than with the
    // site. So queries are embedded and not kept.
    async function embed(text, { signal, remember = false } = {}) {
        const key = remember && cache
            ? embeddingKey({ provider: model.provider ?? 'unknown', model: model.modelId ?? '<embedding>', dim, text })
            : null
        if (key) {
            const hit = cache.get(key, dim)
            if (hit) {
                reused++
                return hit
            }
        }
        const { embedding } = await aiEmbed({ model, value: text, abortSignal: signal })
        // Not `vector`: that is this module's own exported plugin factory, and
        // shadowing it inside the one function that asks the model for money
        // is a trap for whoever edits this next.
        const computed = new Float32Array(embedding)
        if (key) {
            cache.set(key, computed, {
                provider: model.provider ?? 'unknown',
                model: model.modelId ?? '<embedding>',
                dim,
            })
            embedded++
        }
        return computed
    }

    async function findSimilar(storeName, text, { limit = 5, signal } = {}) {
        if (!stores[storeName]) {
            throw new Error(`Unknown vector store: ${storeName}`)
        }
        const vec = await embed(text, { signal })
        return store.findSimilar(storeName, vec, limit)
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

        // Embedding model: a Vercel AI SDK embedding-model factory return,
        // passed by the user. Provider-agnostic by design — users `npm
        // install @ai-sdk/openai` (or anthropic / mistral / cohere / etc.)
        // and pass the model factory through:
        //
        //   vector({ model: openai.embedding('text-embedding-3-small') })
        //
        // We don't carry the provider package here; the SDK kernel ('ai')
        // is enough for embed() to dispatch through the model's own spec.
        model = config.model
        if (!model || typeof model.doEmbed !== 'function') {
            throw new Error(
                'Vector plugin requires a `model` option — a Vercel AI SDK ' +
                'embedding model. Example:\n' +
                "    import { openai } from '@ai-sdk/openai'\n" +
                "    vector({ model: openai.embedding('text-embedding-3-small'), stores: {...} })"
            )
        }

        // Probe dimension via one sample embedding. sqlite-vec needs the
        // exact dim at vec0-table creation; rather than asking the user
        // to maintain a separate `dim` config that has to match the
        // model's actual output, we ask the model. One round-trip at
        // startup; reusable across every embed afterward.
        // This probe is a live call to the embedding provider, so every way a
        // network service can fail arrives here: no credential, a rejected
        // one, a rate limit, an outage. None of those are reasons to fail a
        // static build. The site does not need a search index to render, and
        // taking the whole build down over a missing key means a contributor
        // without provider access cannot build the site at all.
        //
        // So it reports a FAULT and leaves the plugin dormant. `store` stays
        // undefined, which onBeforeRender already reads as "not indexing", and
        // findSimilar / the HTTP route / the MCP tool below are never
        // registered — an agent gets "no such tool" rather than an empty
        // result set that looks like an answer.
        let probe
        try {
            probe = await aiEmbed({ model, value: 'mikser' })
        } catch (err) {
            logger.error(
                {
                    code: 'vector-embedding-unavailable',
                    provider: model.provider ?? 'unknown',
                    model: model.modelId ?? '<embedding>',
                },
                'Vector search is disabled: the embedding model could not be reached (%s). '
                + 'The build is otherwise unaffected — pages render normally — but nothing is '
                + 'indexed this run and findSimilar is not available.',
                err.message)
            return
        }
        dim = probe.embedding.length

        // Opened before the store, and deliberately not through the engine's
        // database module: `createSqliteDatabase` carries the version and
        // config fingerprints that decide when a cache is stale, and being
        // thrown away on a config edit is the one thing this file must never
        // do.
        cache = createEmbeddingCache({
            filename: resolveCachePath({
                cache: config.cache,
                runtimeFolder: runtime.options.runtimeFolder,
            }),
            logger,
        })

        const db = useDatabase()
        if (!db?.isOpen) {
            throw new Error(
                'Vector plugin requires the engine database to be open before its onLoaded fires. ' +
                'This indicates a hook ordering bug — file an issue.'
            )
        }
        store = await createStore({
            db,
            dim,
            stores,
            registerSchema,
        })

        logger.info(
            'Vector store initialized: %s (provider=%s, model=%s, dim=%d, stores=[%s])',
            store.describe(),
            model.provider ?? 'unknown',
            model.modelId ?? '<embedding>',
            dim,
            Object.keys(stores).join(', ') || '<none>'
        )

        logger.info('Vector embedding cache: %d embedding(s) at %s', cache.count(), cache.path)

        // `--clear` is NOT honoured here any more, and there is nothing left
        // to honour it with.
        //
        // This used to call store.clear() on --clear "so the upcoming CREATEs
        // re-embed everything". Both halves were wrong. The clear was already
        // a no-op: the vec0 and ids tables live in mikser.sqlite, and the
        // engine unlinks that whole file before any plugin's onLoaded runs,
        // so there were no rows left to delete. And re-embedding everything is
        // not a thing to arrange on purpose — it is a paid API call per
        // entity, which is what the cache above now stands in front of.
        //
        // What --clear means for this plugin is: the vec0 index is gone and
        // will be rebuilt from the cache, at the price of sqlite inserts.
        // --clear-vectors is the button that spends money, and it says so.

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

            // POST /vector/:storeName  body: { q, limit }
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
            // Public, non-streaming search endpoint. Auth is per-store
            // (a store may carry a token, enforced in-handler); the mount
            // itself isn't loopback-gated, so a facade should proxy it.
            registerRoute({
                path:        base,
                plugin:      'vector',
                reachability: 'public',
                streaming:   false,
                label:       'Vector search',
                displayPath: `${base}/:storeName`,
            })
        }
    })

    // Tool registration. Registers against the engine's tool registry,
    // which exists whether or not the mcp plugin is installed — so there is
    // no gate to write and no ordering rule to obey. The engine stores the
    // BARE name and mcp prefixes on the way out, so a session still sees
    // mikser_vector_find_similar. Lives in the vector plugin (not core) per
    // ADR-0006 — domain logic owned by the plugin, the substrate in core.
    onLoaded(() => {
        const logger = useLogger()
        registerTool('vector_find_similar',
            {
                description: 'Semantic search over a configured vector store. Returns the top-N items closest to the query in embedding space, each with its original mapped data attached. Use this to answer "find pages similar to X" or "which entities are about Y" where exact filtering via mikser_query_entities would miss synonyms, translations, and paraphrases. Available store names are configured under vector.stores in mikser.config.js and visible in the mikser://config resource.',
                inputSchema: {
                    store: z.string().describe('Vector store name. Configured under vector.stores in mikser.config.js. Read mikser://config to discover what stores exist.'),
                    query: z.string().describe('Free-text query. Embedded via the configured AI-SDK embedding model (OpenAI / Anthropic / Mistral / etc.) and matched against the store via cosine similarity.'),
                    limit: z.number().int().min(1).max(50).optional().describe('Max results to return. Default 5, capped at 50.'),
                },
            },
            async ({ store, query, limit = 5 }) => {
                const ok = (data) => ({
                    content: [{ type: 'text', text: JSON.stringify(data, null, 2) }],
                })
                const fail = (msg) => ({
                    isError: true,
                    content: [{ type: 'text', text: msg }],
                })
                try {
                    if (!stores[store]) {
                        const known = Object.keys(stores).join(', ') || '<none configured>'
                        return fail(`Unknown vector store: ${store}. Available: ${known}.`)
                    }
                    const effectiveLimit = Math.min(50, Math.max(1, limit))
                    const results = await findSimilar(store, query, { limit: effectiveLimit })
                    return ok({
                        store,
                        query,
                        limit: effectiveLimit,
                        count: results.length,
                        results,
                    })
                } catch (err) {
                    logger.error('MCP mikser_vector_find_similar error: %s', err.message)
                    return fail(err.message)
                }
            },
        )

        logger.debug('MCP tool registered: mikser_vector_find_similar (vector plugin)')
    })

    onBeforeRender(async (signal) => {
        const logger = useLogger()

        // --clear-vectors: forget every embedding and pay for it again.
        //
        // Read per cycle rather than in onLoaded, so a running watcher can be
        // asked for it: onLoaded runs once per process, and a flag read there
        // is accepted over the socket, reaches nothing and reports success.
        // Taken off runtime.options once acted on, so `--watch
        // --clear-vectors` empties the cache on its first cycle rather than
        // on every save for the life of the process.
        //
        // It empties the cache and nothing else. Re-embedding is what the
        // CYCLE does, and a cycle only reaches the entities it was given —
        // which, on a build where nothing changed, is none of them. So the
        // line says what actually happens rather than what the flag's name
        // suggests: pair it with --clear, and every entity arrives as a
        // CREATE, misses the empty cache, and is embedded again.
        //
        // The index is deliberately left alone. Emptying it here would take
        // search down until something re-indexed, which on a quiet watcher is
        // never.
        if (runtime.options.clearVectors) {
            runtime.options.clearVectors = false
            if (!cache) {
                logger.warn('--clear-vectors: nothing to clear, the embedding model was '
                    + 'unavailable at startup and this plugin is dormant')
            } else {
                const removed = cache.clear()
                logger.info('Vector embedding cache cleared: %d embedding(s). Entities are embedded '
                    + 'again as they are indexed; run with --clear to re-index all of them', removed)
            }
        }

        if (!store || Object.keys(stores).length === 0) return

        embedded = 0
        reused = 0

        // Per-store concurrency for OpenAI embedding calls. Low default
        // (4) keeps us comfortably under the default rate limits; bump via
        // vector.concurrency or per-store.concurrency for larger budgets.
        const globalConcurrency = config.concurrency ?? 4

        for (const storeName of Object.keys(stores)) {
            const { map, pick } = stores[storeName]
            const query = stores[storeName].query ?? (entity => entity.type === 'document')
            if (typeof query !== 'function') {
                logger.warn('Vector store %s has invalid query (not a function); skipping', storeName)
                continue
            }
            const concurrency = stores[storeName].concurrency ?? globalConcurrency

            await pMap(
                useJournal(
                    `Vector ${storeName}`,
                    [OPERATION.CREATE, OPERATION.UPDATE, OPERATION.DELETE],
                    signal,
                ),
                async ({ operation, entity }) => {
                    if (!query(entity)) return

                    if (operation === OPERATION.DELETE) {
                        await store.delete(storeName, entity.id)
                        logger.trace('Vector deleted %s: %s', storeName, entity.id)
                        return
                    }

                    const result = await entityToEmbedding(entity, { map, pick })
                    if (!result) {
                        logger.trace('Vector skip %s — no text: %s', storeName, entity.id)
                        return
                    }

                    try {
                        const vec = await embed(result.text, { signal, remember: true })
                        await store.upsert(storeName, entity.id, vec, result.data)
                        logger.debug('Vector embedded %s: %s', storeName, entity.id)
                    } catch (err) {
                        // Swallow per-entity failures so one bad embedding
                        // (rate limit, transient network) doesn't poison
                        // the whole batch. The journal entry stays in
                        // CREATE state so the next cycle retries.
                        logger.error('Vector embed error %s %s: %s', storeName, entity.id, err.message)
                    }
                },
                { concurrency, signal },
            )
        }

        // Only when something happened. An idle watch cycle has nothing to
        // say, and a line per save is how a useful count becomes noise.
        //
        // Both numbers, because the reused one is the whole point: after a
        // cache wipe this reads `0 embedded, 11003 reused`, which is the
        // difference between rebuilding an index and buying it again.
        if (embedded || reused) {
            logger.info('Vector: %d embedded, %d reused from the cache', embedded, reused)
        }
    })
        // Names this package to the runtime's loaded-plugin record, so
        // ping reports it as running rather than as undetectable.
        return { module: import.meta.url }
    }
}
