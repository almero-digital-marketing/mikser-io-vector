import OpenAI from 'openai'
import _ from 'lodash'
import pMap from 'p-map'
import { encode as toonEncode } from '@toon-format/toon'

// Driver selection. `client` follows the knex client naming convention
// so users can write what they're used to. We dynamic-import the chosen
// driver module so installs that only use sqlite never load pg, and
// vice versa.
const DRIVERS = {
    'better-sqlite3': './src/drivers/sqlite.js',
    'sqlite': './src/drivers/sqlite.js',
    'sqlite3': './src/drivers/sqlite.js',
    'pg': './src/drivers/postgres.js',
    'postgres': './src/drivers/postgres.js',
    'postgresql': './src/drivers/postgres.js',
}

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

    let driver
    let openai
    let model
    let dim

    async function embed(text) {
        const { data } = await openai.embeddings.create({
            model,
            input: text,
            dimensions: dim,
        })
        return new Float32Array(data[0].embedding)
    }

    async function findSimilar(storeName, text, { limit = 5 } = {}) {
        if (!stores[storeName]) {
            throw new Error(`Unknown vector store: ${storeName}`)
        }
        const vec = await embed(text)
        return driver.findSimilar(storeName, vec, limit)
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

        const clientName = config.client ?? 'better-sqlite3'
        const driverPath = DRIVERS[clientName]
        if (!driverPath) {
            throw new Error(
                `Unknown vector.client "${clientName}". ` +
                `Supported: ${Object.keys(DRIVERS).join(', ')}`
            )
        }
        const { createDriver } = await import(driverPath)
        driver = await createDriver({
            runtime,
            dim,
            stores,
            connection: config.connection,
            logger,
        })

        logger.info(
            'Vector store initialized: %s (model=%s, dim=%d, stores=[%s])',
            driver.describe(), model, dim,
            Object.keys(stores).join(', ') || '<none>'
        )

        // Honor `--clear`: wipe every store's rows so the upcoming
        // CREATEs (mikser also clears its catalog) re-embed everything.
        // Critical for the pg backend whose tables live remotely and
        // aren't touched by the runtimeFolder wipe.
        if (runtime.options.clear && Object.keys(stores).length) {
            await driver.clear()
            logger.info('Vector store cleared: [%s]', Object.keys(stores).join(', '))
        }

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
            logger.info('Vector search mounted: %s/:storeName', base)
        }
    })

    onBeforeRender(async (signal) => {
        const logger = useLogger()
        if (!driver || Object.keys(stores).length === 0) return

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
                        await driver.delete(storeName, entity.id)
                        logger.trace('Vector deleted %s: %s', storeName, entity.id)
                        return
                    }

                    const result = await entityToEmbedding(entity, { map, pick })
                    if (!result) {
                        logger.trace('Vector skip %s — no text: %s', storeName, entity.id)
                        return
                    }

                    try {
                        const vec = await embed(result.text)
                        await driver.upsert(storeName, entity.id, vec, result.data)
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
    })
}
