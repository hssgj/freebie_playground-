# Verification report — PAE Context Guard v1

All commands below were run in this workspace on 2026-11-03 (UTC). Everything is
offline: deterministic scripted providers, a loopback HTTP fixture, and the
repository's own echo/fixture assembly harness. **No Qwen, Groq or Gemini
endpoint was contacted; no API key was read or used.**

## Baseline

| Item | Value |
|---|---|
| Repository | `https://github.com/hssgj/PAE` |
| Baseline commit | `0bffa8d164d001b50f68175a5fcd9625efb6bbae` |
| Baseline subject | `Log November 1 morning orientation and bounded inbox review` |
| Python | 3.12.3 |
| Test runner | pytest 9.2.0 (installed into a sandbox-local target dir; the image's site-packages is not writable). The new tests also run under stdlib `unittest`. |

### Baseline results (before any change)

```
$ python3 -m pytest prototype_v0 -q
......................................................          [100%]
54 passed, 9 subtests passed in 0.36s

$ python3 tools/verify_assembly.py
passed: true, 6 checks
```

One **pre-existing, environment-only** failure appears when the baseline is
checked out *detached* (as a bare `git checkout <sha>` leaves it):

```
FAILED prototype_v0/test_mobile_runtime_identity.py::test_runtime_identity_exposes_safe_version_and_tools
E  assert info["branch"]
E  AssertionError: assert ''
```

`mobile_server` reads the branch with `git branch --show-current`, which is empty
on a detached HEAD. On a named branch at the same commit the suite is fully
green (54 passed, shown above). This is not caused by the change and does not
appear in a normal checkout; it is reported here rather than hidden.

## After the change

```
$ python3 -m pytest prototype_v0 -q
80 passed, 9 subtests passed in 2.94s          # 54 existing + 26 new

$ cd prototype_v0 && python3 -m unittest test_context_guard
Ran 26 tests in 2.416s
OK                                             # exit 0, no third-party deps

$ python3 tools/verify_assembly.py
passed: true
checks: catalog_lists_without_credentials_or_session,
        cli_input_provider_agent_source_and_session,
        cli_restart_preserves_history_and_source,
        profile_selection_existing_http_provider_same_session,
        mobile_http_to_shared_agent_and_tool_registry,
        mobile_process_restart_preserves_history
real_llm_inference: false
phone_installation_verified: false
credentials_used: false
```

## Test suite (26 tests) and requirement coverage

| Requirement | Tests |
|---|---|
| 1. Every call checked, incl. continuations, retries, extraction; CLI + mobile | `test_protocol_repair_also_passes_through_the_guard` (2 provider calls, 2 guard checks), `test_tool_loop_checks_again_on_second_continuation` (3 calls, 3 checks), `test_tool_loop_continuation_is_checked_after_tool_result`, `test_cli_reduces_long_history_and_keeps_session`, `test_mobile_reduces_long_history_and_reports_budget` |
| 2. Reserve real max output + margin; precedence; report budget; never change capacity | `test_small_request_passes_unchanged`, `test_profile_declaration_wins_over_launcher_environment`, `test_launcher_environment_used_without_profile`, `test_provider_max_tokens_is_reserved_on_legacy_path`, `test_margin_is_a_safety_knob_not_capacity`, `test_different_capacities_and_output_reservations_change_the_budget`, `test_same_request_fits_bigger_capacity_but_not_smaller`, `test_invalid_or_impossible_budget_fails_loudly` |
| 3. Reduce oldest complete exchanges; keep current request and active tool exchange | `test_long_history_reduced_while_current_request_survives`, `test_active_tool_exchange_is_preserved`, `test_history_with_tool_exchange_is_dropped_atomically` |
| 4. Preserve instructions/permissions/confirmations and required sources | asserted inside `test_long_history_reduced_while_current_request_survives` (governing prompt, tool protocol, `CANON-ONE` source survive) and the atomic-drop test (no orphaned tool result) |
| 5. Required material too large → stop before network, useful error, no retry | `test_required_sources_too_large_causes_zero_provider_calls`, `test_preamble_alone_oversized_names_the_real_blocker`, `test_state_extraction_rejects_oversized_source_bundle_clearly`, `test_cli_refuses_oversized_preamble_with_zero_requests` (fixture received **0** requests), `test_mobile_refuses_required_material_with_zero_requests` (fixture received **0** requests) |
| 6. Persisted history/sources/state/pending intact; work on copies | `test_persisted_session_file_unchanged_by_reduction` (session file byte-identical, 20 messages, derived state and pending action preserved), `test_reduction_never_mutates_the_caller_list`, mobile test asserts the session keeps all 26 messages |
| 7. Estimate labelled, limitations documented | `test_estimate_is_labelled_and_derived_from_characters`, README *Limitations* |
| 8. Concise diagnostics, no contents or credentials | `test_diagnostics_report_counts_without_contents` (no message text in the dict), mobile `/api/runtime` assertions on capacity/reservation/omitted counts |
| State extraction happy path still works | `test_state_extraction_works_when_bundle_fits` |

Integration tests spawn **real processes**: `python app.py` (CLI) and
`mobile_server.py` in a temporary copy of the tree, both against a loopback HTTP
fixture that counts requests and asserts the payload (`max_tokens`, last user
message, absence of the dropped oldest exchange).

## Are the tests meaningful? (mutation check)

The guard was removed from `agent_loop._guarded_chat` and
`state_extractor.extract_state` in a throwaway copy, and the new suite was run
unchanged:

```
$ cd /tmp/mutant && python3 -m pytest prototype_v0/test_context_guard.py -q
9 failed, 17 passed
```

So 9 tests genuinely depend on the guard being wired in; the remaining 17 cover
budget resolution and reduction logic directly.

## Patch verification

`integration.patch` carries the 7 runtime files (new `context_guard.py` plus the
six modified files); the new test file is delivered separately as
`prototype_v0/test_context_guard.py` because it changes no runtime behaviour.

```
$ git clone https://github.com/hssgj/PAE /tmp/patchtest
$ cd /tmp/patchtest && git checkout 0bffa8d164d001b50f68175a5fcd9625efb6bbae
$ git apply --check integration.patch   →  CLEAN
$ git apply integration.patch           →  OK
cmp of all 7 runtime files vs the authored tree → all identical
$ cp test_context_guard.py prototype_v0/
$ python3 -m pytest prototype_v0 -q                    →  80 passed, 9 subtests passed
$ cd prototype_v0 && python3 -m unittest test_context_guard  →  Ran 26 tests, OK (exit 0)
$ python3 tools/verify_assembly.py                     →  passed: true, 6 checks
```

Patch size: 25,617 bytes / 650 lines, 7 hunks-bearing file sections, applied to
the exact baseline with no fuzz and no rejects. Because the run was on a named
branch, the detached-HEAD `branch` assertion described under *Baseline* does not
appear here.

## Behaviour observed

CLI startup on the default (undeclared) path:

```
[context] capacity=4096 output=512 margin=384 input_budget=3200 (capacity=default; output=default; margin=default; tokens are an estimate)
```

A reduced turn (real CLI run, 12 persisted exchanges, fixture capacity 3000):

```
[context guard] chat: estimated_input=2504 <= input_budget=2516 omitted_exchanges=8 omitted_messages=16 (capacity=3000 output=100 margin=384; estimate, not tokenizer-exact)
```

A refusal (zero provider requests reached the fixture):

```
[agent error] Request refused before the provider call: required material does not fit.
Purpose: chat
Estimated input: 2139 tokens (available input budget 616 = capacity 1200 - output 200 - margin 384).
Required material: tool_protocol 1955 tokens (1 message(s)), ...
Nothing was sent, no model was switched and no retry was attempted. What needs narrowing: the fixed runtime preamble ... alone needs ~2122 tokens, more than the whole input budget of 616. This declared capacity cannot host the registered tool protocol: ...
```

## Known limitations of this evidence

1. **Offline only.** Deterministic scripted providers and a loopback fixture. No
   real model was called, so this does not show that Qwen/Groq/Gemini accept or
   behave well under the reduced requests. `real_llm_inference: false` is
   accurate for every result above.
2. **Estimates, not tokenizer counts.** `ceil(chars/4) + 4` per message. It is
   reproducible and dependency-free, but not exact, and it undercounts dense
   content (Czech, CJK, JSON, code). The margin is the only buffer; server-side
   acceptance is not guaranteed.
3. **`local-qwen` (2048 / 512) refuses every turn** by design, because the
   shipped 9-tool preamble is ~2.1k estimated tokens against a 1152-token input
   budget. Documented in the README with remedies; no declared capacity was
   changed to hide it.
4. **`tools/verify_assembly.py` fixture capacity raised 1024 → 8192.** A
   1024-token fixture cannot host the runtime preamble, so the transport
   assertion would have been refused. All of the harness's assertions are
   unchanged.
5. **The test file is not inside `integration.patch`.** It is delivered at its
   exact PAE-relative path and must be copied in (or extracted from the
   package ZIP). Every runtime file is in the patch.
6. **No phone validation, and no recovery of the earlier phone-local guard.**
   The old implementation was unavailable; this is an independent
   implementation against the repository baseline.
7. **Not exercised:** live Gmail/GitHub tool writes (unchanged behaviour, mocked
   upstream), Android APK, concurrent multi-client mobile conversations.
