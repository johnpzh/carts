# ARTS Dialect

ARTS is the abstract ARTS-machine dialect. It materializes CODIR codelets and
SDE/CODIR plan facts into DB, EDT, dependency, epoch, and resource-binding
objects without exposing runtime ABI calls.

ARTS must not rediscover source semantics, owner dims, dependency-window
legality, or codelet captures.

Primary docs:

- [`analysis.md`](./analysis.md)
- [`optimizations.md`](./optimizations.md)
