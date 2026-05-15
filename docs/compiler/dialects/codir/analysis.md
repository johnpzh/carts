# CODIR Analyses

CODIR analyses operate after SDE planning and before ARTS object creation.

Owned analyses:

- codelet capture analysis: verifies that every external value is a dep, param,
  local value, or value derived from them;
- dep/param ABI analysis: classifies memory deps, control deps, scalar params,
  yielded values, and illegal mutable params;
- token-local access analysis: verifies codelet load/store indices against MU
  token-local views;
- codelet-local effect analysis: proves local-only scratch, scalar forwarding,
  and token access modes;
- launch-shape analysis: checks that logical launch operands match the isolated
  body ABI.

Emitted facts:

- explicit dep list;
- explicit param list;
- token-local view operands and shapes;
- yielded values and completion edges;
- codelet-local effect summaries;
- diagnostics for implicit captures or unsupported token-local rewrites.

CODIR analysis facts lower to ARTS object operands and attrs. ARTS should not
rerun CODIR capture analysis.
