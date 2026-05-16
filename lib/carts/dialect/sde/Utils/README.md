# SDE Utils Implementations

Implement SDE-only utility helpers here when more than one SDE analysis,
transform, conversion, or verifier needs the same source-semantic helper.

Keep one-pass logic pass-local when it does not express an SDE invariant and is
not reused by a sibling SDE component.
