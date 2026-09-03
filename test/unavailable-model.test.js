// A missing or rejected credential must not fail the build.
//
// The plugin probes the embedding provider once at init to learn the vector
// dimension. That probe is a live network call, so it fails for every reason a
// network service fails: no API key, a rejected one, a rate limit, an outage.
// None of those are reasons a static site cannot be built — the pages do not
// need a search index to render — and taking the build down over one means a
// contributor without provider access cannot build the site at all.
//
// So the probe failure is reported as a FAULT (logger.error carrying a `code`,
// which the engine surfaces in --json and mikser_ping without touching the
// exit code) and the plugin goes dormant.

import { describe, it } from 'node:test'
import assert from 'node:assert/strict'

import { vector } from '../index.js'

// Minimal Vercel AI SDK embedding-model shape. `doEmbed` is what the plugin
// checks for, and what the SDK kernel eventually calls — throwing there is
// exactly what a missing key does one layer down.
function failingModel(message = 'OpenAI API key is missing.') {
    return {
        specificationVersion: 'v2',
        provider: 'openai',
        modelId: 'text-embedding-3-small',
        maxEmbeddingsPerCall: 1,
        supportsParallelCalls: false,
        async doEmbed() { throw new Error(message) },
    }
}

// Captures the hooks the plugin registers plus everything it logged, so a test
// can run init and then inspect both.
function harness(model) {
    const hooks = {}
    const logged = { error: [], warn: [], info: [] }
    const logger = {
        error: (...a) => logged.error.push(a),
        warn:  (...a) => logged.warn.push(a),
        info:  (...a) => logged.info.push(a),
        debug: () => {}, trace: () => {}, notice: () => {},
    }
    const runtime = { options: {} }

    // The plugin registers onLoaded more than once; keeping only the last
    // silently drops the init hook, so collect them and run them in order the
    // way the engine does.
    const loaded = []
    const descriptor = vector({ model, stores: { documents: {} } })
    hooks.loaded = async () => { for (const fn of loaded) await fn() }
    descriptor({
        runtime,
        onLoaded:        fn => { loaded.push(fn) },
        onBeforeRender:  fn => { hooks.beforeRender = fn },
        onProvision:     () => {},
        useLogger:       () => logger,
        useJournal:      () => [],
        useDatabase:     () => { throw new Error('database must not be touched when the probe failed') },
        registerSchema:  () => {},
        registerRoute:   () => {},
        registerTool:    () => {},
        constants: { OPERATION: { CREATE: 'create', UPDATE: 'update', DELETE: 'delete' } },
    })
    return { hooks, logged, runtime }
}

describe('embedding provider unavailable', () => {
    it('does not throw out of init when the probe fails', async () => {
        const { hooks } = harness(failingModel())
        await assert.doesNotReject(
            () => hooks.loaded(),
            'a missing credential must not propagate out of onLoaded — that is what killed the build',
        )
    })

    it('reports it as a fault, with a code and the provider message', async () => {
        const { hooks, logged } = harness(failingModel('OpenAI API key is missing.'))
        await hooks.loaded()

        assert.equal(logged.error.length, 1, 'expected exactly one fault')
        const [fields, message] = logged.error[0]
        assert.equal(fields.code, 'vector-embedding-unavailable',
            'the code is what makes this a fault rather than an error line')
        assert.equal(fields.provider, 'openai')
        assert.equal(fields.model, 'text-embedding-3-small')
        assert.match(message, /disabled/)
        // The provider's own message is the actionable part — "API key is
        // missing" tells the reader what to do; "probe failed" does not.
        assert.ok(logged.error[0].some(a => typeof a === 'string' && /API key is missing/.test(a)),
            `the provider message must survive into the fault\n${JSON.stringify(logged.error[0])}`)
    })

    it('goes dormant rather than half-initialized', async () => {
        const { hooks, runtime } = harness(failingModel())
        await hooks.loaded()

        // findSimilar absent, not present-and-returning-nothing: an empty
        // result set reads as "no matches", which is the one answer a broken
        // search must never give.
        assert.equal(runtime.findSimilar, undefined,
            'findSimilar must not be exposed when nothing was indexed')

        // And indexing is a no-op. useDatabase throws in the harness, so if
        // this reached the store at all the test fails loudly.
        await assert.doesNotReject(() => hooks.beforeRender(),
            'the render hook must skip cleanly when init did not complete')
    })
})
