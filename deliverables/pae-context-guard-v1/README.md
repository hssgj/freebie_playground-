# PAE Context Guard v1

A request-budget guard for the existing PAE runtime. It checks the complete
working request before every real provider call, reduces only old persisted
conversation history, and refuses — before the network call — when required
material cannot fit.

- **Source baseline (exact):** `hssgj/PAE` @ `0bffa8d164d001b50f68175a5fcd9625efb6bbae`
- **Integration form:** `integration.patch` (applies cleanly to that commit), plus the
  delivered files in their PAE-relative paths.
- **Tests:** `prototype_v0/test_context_guard.py` (26 tests, stdlib `unittest`, no network).
- **Verification evidence:** [VERIFICATION.md](VERIFICATION.md)

The runtime, persistent sessions, mobile bridge and startup model profiles were
extended, not replaced. No launcher, session schema, tool authorization rule,
UI or database was changed.

## What was wrong

`config/model_profiles.json` declared `context_tokens` per profile, but nothing
enforced a request budget: `run_agent_turn` sent whatever the runtime assembled.
A long session, a large `--load`ed source, a big tool result and the runtime's
own tool protocol could together exceed the model's capacity, and the failure
surfaced only at the provider.

## What the guard does

`prototype_v0/context_guard.py` resolves a budget, estimates the request, and
returns a guarded **copy** of the message list:

```
input_budget = declared capacity - reserved maximum output - safety margin
```

1. **Every call is checked.** `agent_loop._guarded_chat` is the single choke
   point. The first call, every tool-loop continuation and every protocol repair
   re-enter it, so a request that grows after a tool result is re-checked before
   it is sent. `state_extractor.extract_state` checks its source bundle the same
   way before its own provider call. CLI and mobile share `run_agent_turn`, so
   both get identical protection.
2. **Capacity is declared, not invented.** The reservation is the profile's
   `max_output_tokens` — the same value the provider puts in `max_tokens` — plus
   an explicit margin. A model's declared capacity is never modified.
3. **Only old history is reduced.** When over budget, the oldest *complete*
   exchanges are removed from the working copy, oldest first, until it fits.
   An exchange is everything from one user message up to the next user message,
   so an assistant tool call is never separated from its tool result.
4. **Required material is preserved.** The governing system prompt, the tool
   protocol (permissions and confirmation rules), session facts, history rules,
   all persistent sources, derived state, the latest user request and the active
   tool exchange are never removed or truncated.
5. **Refusal instead of silent truncation.** If required material alone exceeds
   the budget, `ContextBudgetError` is raised before the network call. Nothing
   is sent, no model is switched and no retry is attempted. The error names the
   categories that consume the budget and what needs narrowing — including the
   case where the runtime's own preamble is the blocker.
6. **The archive is untouched.** The guard works on copies of the message dicts
   and never mutates `session.messages`, `session.sources`,
   `session.derived_state` or `session.pending_actions`. The session file is
   byte-identical after a reduced turn (tested).
7. **Counts are an estimate.** See *Limitations*.
8. **Diagnostics carry numbers, not contents.** Capacity, reservation, margin,
   input budget, estimated input, omitted exchange/message counts, origins.

### How cached sources are handled

Sources are loaded into the request as `PERSISTENT SOURCE MATERIAL` system
messages built from the session archive by `app.build_messages`. The guard
treats them as **required material**: they are never dropped, never truncated,
and never replaced by a summary or a pointer to the cache. If they do not fit,
the turn is refused and the message says sources must be unloaded or split. The
cached copy in the session file is never modified by the guard, so the archive
remains authoritative for the next turn.

## Configuration and precedence

| Setting | 1st | 2nd | 3rd | Fallback |
|---|---|---|---|---|
| capacity | profile `context_tokens` | `PAE_CTX_TOKENS` | — | `4096` |
| output reservation | profile `max_output_tokens` | provider `max_tokens` | `PAE_MAX_TOKENS` | `512` |
| safety margin | `PAE_CTX_MARGIN` | — | — | `384` |

- A **selected profile's declaration always wins** for capacity and output, so
  `PAE_CTX_TOKENS`/`PAE_MAX_TOKENS` cannot silently replace it. Those two
  variables apply on the existing launcher path, which starts PAE without
  `--profile` (for example `start-groq.sh`, which already exports
  `PAE_CTX_TOKENS=8192` and `PAE_MAX_TOKENS=1024`).
- `PAE_CTX_MARGIN` always applies. The margin is a safety knob, not a model
  capacity, so tightening it from the environment is allowed.
- The fallbacks are guard defaults for an undeclared capacity, not device
  claims. `4096` is the smallest fallback that can host the shipped 9-tool
  preamble with 512 output and the default margin.
- A malformed value (`PAE_CTX_TOKENS=lots`) raises a clear configuration error
  instead of silently falling back.

Resolved budget is reported once at CLI startup:

```
[context] capacity=4096 output=512 margin=384 input_budget=3200 (capacity=default; output=default; margin=default; tokens are an estimate)
```

and per reduction, when it happens:

```
[context guard] chat: estimated_input=2504 <= input_budget=2516 omitted_exchanges=8 omitted_messages=16 (capacity=3000 output=100 margin=384; estimate, not tokenizer-exact)
```

Mobile exposes the same numbers at `GET /api/runtime` under `context_budget`
(capacity, reservations, origins, estimate kind, plus `last_request` with the
omitted counts). No message contents or credentials appear in either.

## Integration

```sh
# 1. Apply the runtime patch to a clean checkout of the baseline commit
git clone https://github.com/hssgj/PAE
cd PAE
git checkout 0bffa8d164d001b50f68175a5fcd9625efb6bbae
git apply /path/to/integration.patch

# 2. Add the delivered test file (new path, no conflict with the patch)
cp /path/to/prototype_v0/test_context_guard.py prototype_v0/
```

Alternatively copy every file from this directory over the same tree; the
files here are byte-identical to what the patch produces.

**Patch scope.** `integration.patch` (650 lines, 25,617 bytes) carries every
runtime change: the new `prototype_v0/context_guard.py` plus modifications to
`agent_loop.py`, `app.py`, `busy_indicator.py`, `mobile_server.py`,
`state_extractor.py` and `tools/verify_assembly.py`. The test suite is a new
standalone file at `prototype_v0/test_context_guard.py` and is delivered as a
file rather than inside the patch (it changes no runtime behaviour). Applying
the patch and copying that one file reproduces the delivered tree exactly —
verified in a fresh clone, see VERIFICATION.md.

## Verification

```sh
cd prototype_v0
python -m unittest test_context_guard      # stdlib only, no dependencies
python -m pytest test_context_guard.py -q  # if pytest is available
cd ..
python tools/verify_assembly.py            # existing assembly proof
python -m pytest prototype_v0 -q           # existing regressions + new tests
```

All of the above are offline: deterministic scripted providers and a loopback
HTTP fixture. They are **not** Qwen, Groq or Gemini validation.

## Limitations

1. **Token counts are an estimate, not a tokenizer.** The estimate is
   `ceil(len(content)/4) + 4` per message (`context_guard.CHARS_PER_TOKEN`).
   It is convenient, dependency-free and reproducible, but it is *not*
   tokenizer-exact and it *undercounts* dense content such as Czech prose, CJK,
   minified JSON and code. The safety margin is the only buffer. Nothing here
   guarantees that a provider will accept a request it is sent; the guard bounds
   the runtime's own budget, not a server's accounting.
2. **Only conversation history is reducible.** Governed by design: required
   material is refused rather than shrunk, so there is no summarisation,
   no source trimming and no silent truncation of the current request.
3. **The `local-qwen` profile does not fit the shipped tool protocol.** That
   profile declares `context_tokens: 2048` with `max_output_tokens: 512`, which
   leaves 1152 input tokens with the default margin. The runtime's own preamble
   (9 tool specs + session facts + history rules) is ~2.1k estimated tokens, so
   the guard refuses *every* turn on that profile with an explicit error rather
   than oversending. This is requirement-5 behaviour, reported honestly, not a
   silent regression: the remedies are to start `llama-server` with a larger
   `-c` and raise the declared capacity to match, or to register fewer tools.
   No profile value was changed here.
4. **Launchers were not modified.** `start-local.sh` / `start-mobile.sh` start
   `llama-server -c 2048` without exporting `PAE_CTX_TOKENS`, so on that path the
   guard falls back to 4096. Export `PAE_CTX_TOKENS=2048` there to make the
   declaration truthful (the guard will then refuse, per limitation 3).
5. **`tools/verify_assembly.py` fixture capacity was raised** from `1024` to
   `8192`. A 1024-token model cannot host the runtime preamble, so the harness's
   transport assertion would have been refused before the request. Its actual
   assertions (model id, `max_tokens=77`, no auth header, same session) are
   unchanged; nothing was removed or weakened.
6. **The echo profile is guarded too**, using the fallback/env capacity. Echo
   makes no network call, so this only exercises the code path.
7. **No phone validation.** Nothing here was run on the phone, and the earlier
   phone-local `context_guard.py` was not available and was not recovered or
   reused. This is an independent implementation against the repository
   baseline.

## Licence and scope

Same repository terms as `hssgj/PAE`. The canonical repository and the phone
were not modified; this work was delivered to `hssgj/freebie_playground-` for
independent review.
