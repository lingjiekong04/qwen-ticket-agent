# Offline integration release candidate v1

## Outcome

- Nonlive acceptance: **PASS**
- Runtime: `trusted_completion_v1.7`
- Scenarios: `11/11`
- Tools covered: `8/8`
- Recovery paths: `3/3`
- Rollback rehearsal: `PASS`
- Backend: `in_memory_only`
- Model/GPU/network/live tools: `not used`

## Frozen composition

`Qwen2.5-7B-Instruct + retention-v1.2 LoRA + trusted_completion_v1.7`

This is an offline integration release candidate, not a live deployment. The compatibility replay does not alter or re-score the consumed v6 holdout.

## Rollback

Rollback is an atomic switch to `manual_safe_stop_v1`. New requests are held for manual review with zero backend events. The formally failed v1.6 runtime is retained for comparison only and is not an approved fallback.
