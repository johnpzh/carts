# Regression-Guard Checklist

Run this checklist after every fix, before marking the task complete and advancing to the next item. The whole point is to catch regressions in prior-green samples while the change is still small enough to bisect.

## The 10 items

Run in order. Stop at the first failure and investigate.

1. **Recompile.** `dekk carts build` (full rebuild if the change touches common utilities or TableGen). Verify clean compile.

2. **Per-dialect lit tests.** `dekk carts lit lib/carts/dialect/{sde,codir,arts,arts-rt}/test/` filtered to the touched pass(es). All pass. Metadata-only fixtures may not exercise OpenMP conversion; do not treat that as a regression by itself.

3. **Pass-test suite.** `dekk carts test`. Full pass-test suite (fast). No new failures.

4. **Re-run a prior-green sample (single-node).** Pick one sample that was passing before this change. `dekk carts compile samples/<sample> -O3 -o .carts/outputs/regression/<sample> && .carts/outputs/regression/<sample>`. Output matches baseline.

5. **Sample suite snapshot.** `dekk carts test --suite e2e --json .carts/outputs/regression/carts-e2e.json`. Compare PASS count to the latest baseline in `.results/`. Do not advance if PASS count decreased. New PASS count should match or exceed prior snapshot.

6. **Pipeline-stage IR sanity.** If the fix touches stage N, dump full pipeline: `dekk carts compile samples/<fixed-sample> --all-pipelines -o .carts/outputs/regression/pipelines/`. Run `grep -c "arts\." .carts/outputs/regression/pipelines/*/*.mlir` to spot op-count anomalies. No silent op drops.

7. **Verify-barrier check.** Confirm the verification barrier above the fix still holds. `dekk carts lit lib/carts/dialect/*/test/ -filter=Verify<StageName>`.

8. **Multinode spot-check** (if the fix touches SDE distribution planning, DB refinement, ownership, EDT materialization, EpochLowering, or CPS logic). `dekk carts compile samples/<fixed-sample> --distributed-db -O3 -o .carts/outputs/regression/mn && ARTS_CONFIG=samples/arts_multinode.cfg .carts/outputs/regression/mn`. No regression vs prior multinode baseline.

9. **Stderr scan.** Capture stderr: `dekk carts compile … 2>&1 | grep -i "warn\|error"`. New warnings? Investigate before considering the fix done.

10. **Clean rebuild + re-test.** `dekk carts build --clean && dekk carts test`. Catches stale build state masking the fix or reintroducing old bugs.

## Skip rules

You may skip an item only with explicit reason. Document the skip in the fix-attribution log entry.

- Skip 8 (multinode) if the fix demonstrably cannot affect distribution: doc-only change, comment fix, ARTS-RT-only LLVM-near pass, or pure SDE analysis with no contract change.
- Skip 10 (clean rebuild) for trivial doc/comment changes.
- **Never skip 5 (suite snapshot).** Even doc changes have produced regressions when build artifacts shifted.

## After a green run

1. Append a row to `docs/compiler/fix-attribution-log.md` (create the file on first use):

   ```markdown
   | YYYY-MM-DD | symptom-stage | fix-stage | sample-or-benchmark | one-line rationale | commit-hash |
   ```

2. Mark the task `completed` via `TaskUpdate`.

3. If running under `/loop`, schedule the next iteration with `ScheduleWakeup`. Otherwise return to the user with a short summary.

## After a red run

If the regression-guard caught a regression introduced by your fix:

1. Stop. Do not commit.
2. Identify which item failed.
3. The originating layer for the regression is most likely the same as for the original failure — re-apply the triage rubric.
4. If the fix caused new failures the rubric does not predict, the rubric needs a new row. Add it to `references/triage-rubric.md` "Triage decision tree" so future iterations catch this earlier.
