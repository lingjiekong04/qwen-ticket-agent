# Rollback plan

1. Stop admitting new work to `trusted_completion_v1.7`.
2. Atomically select `manual_safe_stop_v1` in the process-local release registry.
3. Verify that a probe request returns `manual_safe_stop`, creates no backend event,
   and requires manual review.
4. Preserve request/audit digests and the failing input; do not replay writes.
5. Diagnose offline. A new runtime revision needs its own nonlive acceptance before
   it may replace the safe stop.

`exact_create_recovery_v1.6` is not a production rollback target because it failed
the frozen v6 comparison gates. It is retained only for forensic comparison.

