# CARTS Plan Index

The master plan is [`../master-plan.md`](../master-plan.md). This directory
contains implementation subplans that can move independently while preserving a
single architecture contract.

Subplans:

- [`folder-reorganization.md`](./folder-reorganization.md)
- [`subdialect-analysis-optimization.md`](./subdialect-analysis-optimization.md)
- [`utility-ownership.md`](./utility-ownership.md)
- [`dialect-stack-migration.md`](./dialect-stack-migration.md)
- [`codir-edt-isolation.md`](./codir-edt-isolation.md)
- [`memref-mu-token-rewrite.md`](./memref-mu-token-rewrite.md)
- [`arts-materialization-cleanup.md`](./arts-materialization-cleanup.md)
- [`performance-large64.md`](./performance-large64.md)
- [`verification-release.md`](./verification-release.md)

Each subplan should state:

- objective;
- current implementation surface;
- target contract;
- phased work;
- exit gates;
- tests and benchmark evidence.
