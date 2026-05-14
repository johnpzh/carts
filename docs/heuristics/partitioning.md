# Partitioning (Technical Guide)

This document explains CARTS DB partitioning as a technical reference.
It focuses on:
- What each partitioning mode means
- How the data layout changes (ASCII diagrams)
- How index localization works (db_ref vs local indices)
- Where decisions are made (analysis vs heuristics vs rewriting)

Related guide:
- Distribution (H2): `docs/heuristics/distribution.md`
- This document covers partitioning (H1) only.

---

## 0) The Core Distinction: Allocation Layout vs Acquire Range

Partitioning has two orthogonal concepts:

1) Allocation layout (how the DB is shaped and how many DBs exist)
2) Acquire range (how many DB entries a particular acquire requests)

Coarse is a *layout* choice (outer sizes = 1). Full-range is an *acquire* choice
(request all blocks), and can happen under block, stencil, or fine-grained
layouts.

Coarse layout (single DB)
- outerSizes = [1]
- innerSizes = original shape
- there is exactly one DB

Full-range acquire (many DBs)
- outerSizes = many blocks/elements
- acquire offsets = [0]
- acquire sizes = [numBlocks]
- you keep the multi-block layout, but this acquire spans all blocks

Why this matters:
- Coarse: simplest correctness, but forces shared DB access (poor locality and
  less concurrency).
- Full-range: keep the block layout (good concurrency), but allow an access
  pattern that is not tied to the per-task partition offset.

---

## 1) Terminology and Notation

For a logical array A[d0][d1]...[dR-1]:

- partitionedDims: which original dimensions are partitioned (supports
  non-leading for Block / Fine-grained).
- outerSizes: number of DB entries per partitioned dimension, in
  partitionedDims order.
- innerSizes: per-DB shape, in original dimension order (partitioned dims are
  replaced with block sizes or 1 depending on mode).
- startBlock: acquire-specific outer offset (block index) used for localizing
  indices.
- blockSize: the per-chunk size along the partitioned dimension (Block / Stencil).

Index mapping always has two levels:
- db_ref indices: choose which DB entry (outer index tuple).
- local indices: address inside the inner memref for that DB.

## 1.1) Relationship to Distributed DB Ownership

`--distributed-db` is an ownership/routing concern, not a partition
mode.

- H1 partitioning still decides DB layout (`coarse`, `block`, `stencil`,
  `fine_grained`) and acquire localization.
- Distributed ownership is applied after partitioning and marks eligible
  `DbAllocOp`s with `distributed`.
- Lowering then chooses per-block owner routes for marked allocations
  (currently round-robin).
- `--distributed-db` also enables the distributed init path (reserve in
  `initPerNode`, owner-local create in `initPerWorker`) for marked DBs.

So, partitioning decides *shape*; distributed ownership decides *which node owns
each shaped DB entry*.

---

## 2) Responsibility Split (Who Does What)

Partitioning is intentionally split into roles. The rule is:

- SDE `PatternAnalysis`, tiling, and distribution planning decide policy while
  source/tensor facts are visible.
- Core acquire/alloc analyses validate and refine the already-authored DB shape.
- `CreateDbs` bridges raw memrefs by materializing SDE-authored layout and slice
  facts.
- Raw-memref indexers execute the already-chosen layout

Flow of responsibility:

  SDE plan attrs -> CreateDbs -> DbPhysicalLayoutPlan -> raw-memref indexers
  (policy)          (bridge)     (materialized shape)    (mechanical rewrite)

What each layer owns:

- DbGraph / DbAcquireNode: canonical per-acquire validation facts, including access
  classification, mapped partition dims, indirect/direct flags, and cached
  `DbAcquirePartitionFacts`.
- DbAllocNode: aggregation across acquires for one allocation.
- SDE `PatternAnalysis`/distribution/tiling passes: policy. They prove owner
  dims, task grain, access windows, and physical block shape before Core.
- `CreateDbs`: bridge. It converts SDE-authored plan attrs and `DbControlOp`
  slices into `DbPhysicalLayoutPlan`, `arts.db_alloc`, and `arts.db_acquire`.
- Raw-memref indexers: execution. They localize legacy memref loads/stores
  using `PartitionInfo + DbPhysicalLayoutPlan`; they do not choose policy.

Important implementation detail:
- `DbAllocNode` stores direct child acquires in the DB graph.
- `DbPartitioning` does not only look at those direct children; it recursively
  walks nested acquire nodes via `collectAcquireNodesForAlloc()` and
  `collectAcquireNodesRecursive()`.
- So the pass does visit all acquire nodes reachable from one allocation.
- The rest of this guide describes the current implementation model.

### 2.1) The Actual Analysis Objects In Code

Today the partitioning pipeline is built from these layers of state:

1. `DbAcquireNode`
   - Source of truth for one acquire.
   - Owns cached `DbAcquirePartitionFacts`.
   - `DbDimAnalyzer` computes:
     - `DbPartitionEntryFact[]`
     - `DbDimPartitionFact[]`
     - direct / indirect access flags
     - mapped partition dimensions
     - `needsFullRange`
     - `preservesDistributedContract`

2. `DbAnalysis::AcquirePartitionSummary`
   - Returned by `analyzeAcquirePartition(...)`.
   - Carries:
     - `mode`
     - `partitionOffsets`
     - `partitionSizes`
     - `partitionDims`
     - `isValid`
     - `hasIndirectAccess`

3. `AcquirePartitionInfo`
   - `DbPartitioning`'s working copy of the acquire summary.
   - Adds:
     - `accessPattern`
     - `preservesDependencyMode`
     - `needsFullRange`

4. `AcquireInfo` / `PartitioningContext`
   - The heuristic voting layer.
   - Consumes facts already prepared by the graph/pipeline.
   - Does not own canonical legality state.

5. `DbBlockPlanResolver` + `DbPartitionPlanner`
   - Turn the chosen mode into a concrete rewrite plan:
     - `blockSizes`
     - `partitionedDims`
     - allocation shape
     - per-acquire rewrite payloads

### 2.2) Current Canonical Model

The analysis model is now:

```text
per DbAlloc
  -> per DbAcquire
     -> per partition entry
     -> per logical dimension
```

That information lives in `DbAcquirePartitionFacts` on `DbAcquireNode`.
`DbAllocNode` aggregates it. `DbPartitioning` translates it into pass-local
decisions and rewrite payloads.

DB mode note (in/out/inout):
- Db access mode is adjusted by the Db pass based on actual loads/stores.
  "Force inout" style policies should not be needed; if something is read and
  written, it becomes inout automatically.
- Exception: acquires marked with `preserve_access_mode` already carry an
  authoritative dependency contract. We only set this when the compiler already
  knows the exact mode and should not re-check or optimize it later
  (currently explicit `DbControlOp`-derived acquires and worker-local partial
  reduction acquires).

---

## 3) How Decisions Are Taken (Heuristics Overview)

Partitioning decisions happen at two levels:

Allocation-level (one decision for the DbAlloc):
- coarse / fine_grained / block / stencil

Acquire-level (can differ per acquire):
- full-range vs single-block acquire, driven by whether the access depends on
  the partition offset (or is indirect).

### 3.0) The Current Pipeline In Code

The actual pass flow is:

1. Build or rebuild the DB graph.
2. For one `DbAllocOp`, recursively collect all `DbAcquireNode`s.
3. For each acquire, call `DbAnalysis::analyzeAcquirePartition(...)`.
4. Convert that into `AcquirePartitionInfo`.
5. Build `PartitioningContext` for heuristic voting.
6. Ask heuristics for:
   - allocation mode
   - per-acquire `needsFullRange`
7. Reconcile acquire modes.
8. Resolve block sizes and `partitionedDims`.
9. Force safety fallbacks for acquires whose inferred dims disagree with the
   chosen plan.
10. Build `DbPhysicalLayoutPlan` from SDE plan attrs and materialize alloc +
    acquires.

The important point is that the pass is already split into:
- analysis that builds acquire facts
- heuristics that choose allocation mode
- planning/rewriting that applies the decision

The design corresponding to this flow is captured in this guide.

### 3.1 Allocation-Level Heuristics (H1.1-H1.6)

This is the current intent (see
`lib/arts/dialect/core/Analysis/heuristics/PartitioningHeuristics.cpp` and the
`DbHeuristics` wrapper):

- H1.1: Read-only + single-node + no partition capability -> Coarse
- H1.1b: Read-only + single-node + all block-capable acquires are full-range -> Coarse
- H1.2: Direct writes + indirect reads -> Block (indirect uses full-range)
- H1.2b: Direct + indirect writes -> Block (indirect uses full-range)
- H1.3: legacy Core stencil fallback may preserve ESD-style block/window
  contracts when SDE already stamped halo facts, else fallback
  - Special-case: read-only stencil on single-node -> Coarse (avoid halo overhead)
- H1.3b: Double-buffer stencil (jacobi-style) -> Block when all writes are uniform
  - Detects: stencil reads + uniform writes (no cross-element write dependencies)
  - Enables block partitioning for write phases (no halo overhead)
  - Preserves ESD for read phases when needed
- H1.3 (indexed): Indexed patterns -> Block when possible, else element-wise
- H1.4: Uniform direct access -> Block
- H1.5: Multi-node -> Prefer Block (else fine-grained) for network efficiency
- H1.6: Non-uniform + no capability -> Coarse
- H1 (distribution-integrated): if EDT distribution is `tiling_2d` and an
  `inout` acquire carries valid 2D partition sizes, force `blockND` (2D) so
  output ownership matches the distribution plan

Important H1.C3 nuance:
- On a single node, a generic lowering/distribution contract does not by itself
  justify keeping block layout once every block-capable read acquire widened to
  full-range.
- Preserve block here only when the remaining structure still carries real
  value, such as stencil/cross-dimension ownership semantics or multi-node
  placement.
- This matters for cases like `sw4lite/rhs4sg-base`, where a read-only input
  can lose all task-locality benefit even though it still carries block-shaped
  metadata from earlier lowering.

Current H2 tie-in:
- Internode `matmul` is currently selected as `tiling_2d` by distribution
  heuristics, so this `blockND` path is expected to trigger for writable output
  acquires in that case.

Important naming detail:
- In the implementation, "block" is a family: PartitioningDecision::isBlock()
  returns true for both Block and Stencil (Stencil is "Block + ESD halos").

### 3.2 Per-Acquire Full-Range (H1.7)

Some acquires cannot be safely localized to a single block:
- indirect access (data-dependent indices)
- partition offset not present in the access expression

In those cases, the allocation can still be Block/Stencil, but that acquire is
rewritten as full-range (offset=0, size=numBlocks).

Stencil exception:
- Stencil patterns are allowed to omit the partition offset from the access
  expression, because halo handling provides the missing neighbor rows without
  requiring full-range acquires.

`tiling_2d` exception:
- For `tiling_2d` output acquires (`inout`) with explicit 2D owner hints,
  `DbPartitioning` can infer partition dims `[0,1]` and keep those hints even
  when generic dim inference would otherwise be incomplete.
- Read-only inputs in the same loop may still follow regular block localization
  rules; only writable ownership is forced to N-D alignment.

### 3.2.1) Current Safety Checks

Today the safety logic is spread across three places:

1. `DbAcquireNode::getPartitionOffsetDim(...)`
   - proves which access dimension depends on the partition offset
   - returns `std::nullopt` when the mapping is unknown or unsafe

2. `PartitionBoundsAnalyzer::needsFullRange(...)`
   - forces full-range for:
     - indirect access
     - missing offset dependence
     - non-leading read-only stencil accesses

3. `DbPartitioning`
   - after the final block plan is chosen, any acquire whose inferred
     `partitionDims` disagree with the chosen `partitionedDims` is forced to
     full-range

This safety net is necessary, but it is later than ideal. The cleaner design is
to keep per-dimension access proposals all the way into heuristic voting so the
wrong mode is less likely to be chosen in the first place.

### 3.2.2) Mixed-Dimension Design Example

`vel4sg-base` is the canonical example:

- the parallel loop is over `k`
- some helper reads are stencil in `i`
- some helper reads are stencil in `j`
- only `derivative_z` is stencil in `k`

The correct conclusion is:
- partition on `k`
- use block on `k` for arrays that are only stencil in `i/j`
- use stencil-on-`k` only for arrays that actually read `k +/- 1`

That is the shape the redesigned analysis model should represent directly:
- candidate partition dimension = `k`
- per-acquire classification on `k`
- allocation mode chosen from those per-dimension acquire proposals

---

## 3.3) Partitioning Modes At A Glance (Paragraph + Figure)

This section gives one compact paragraph and one compact figure per mode. The
later sections (4-7) keep the full formal detail and worked examples.

### Coarse

Coarse keeps the entire logical array in one DB (`outerSizes=[1]`). There is no
outer block selection during execution: every access uses `db_ref[0]`, and all
indices are local indices into the single inner memref.

```
Logical A[...]  --->  DbAlloc outerSizes=[1], innerSizes=[full shape]
                           |
                           v
                        DbRef[0]
```

### Fine-Grained

Fine-grained creates many small DBs keyed by outer coordinates (often one row
or one element per DB depending on outer rank). Access first selects a DB from
outer indices (`db_ref`), then uses remaining local indices inside that DB.

```
Logical A[i,j] (outerRank=1 on i)
   rows -> DB[0], DB[1], ... DB[D0-1]

  db_ref = i
  local  = j
```

### Block

Block partitions one or more dimensions into contiguous chunks. Each access is
split into block coordinates (`floor(idx/B)`) and local coordinates (`idx%B`).
`db_ref` selects the block; local indices address within that block.

```
A[i,j], block rows by B

  blockIdx = floor(i/B)   -> db_ref[blockIdx]
  localI   = i % B        -> view[localI, j]
```

### Stencil (Block + Halo / ESD)

Stencil keeps Block allocation layout, but dependency delivery is augmented with
halo slices (ESD): each worker sees owned block data plus neighbor boundary
slices (`leftHalo`, `rightHalo`) for ghost-cell accesses.

```
neighbor block   owned block   neighbor block
    |                |               |
    v                v               v
 leftHalo         owned view      rightHalo
 (ESD slice)      (full block)    (ESD slice)
```

### Coarse vs Full-Range (Important)

Coarse is an allocation decision (single DB). Full-range is an acquire decision
(request all DB blocks) and can happen under block/fine/stencil layouts.

---

## 4) Mode 1 - Coarse (Single DB)

Meaning: one DB holds the entire array.

Theorem (Coarse Layout):
If the allocation is coarse, then all indices map directly to the inner memref;
the db_ref index is always 0.

Layout (2D example A[4][8]):

```
Single DB (outerSizes=[1])

  +-------------------------+
  | [0,0]..[0,7]            |
  | [1,0]..[1,7]            |
  | [2,0]..[2,7]            |
  | [3,0]..[3,7]            |
  +-------------------------+
   db_ref[0]
```

Concrete example (2D, A[4][8]):

```
Global A[i,j] (single DB):

  j=0  1  2  3  4  5  6  7
i=0  [0,0][0,1] ...       [0,7]
i=1  [1,0][1,1] ...       [1,7]
i=2  [2,0][2,1] ...       [2,7]
i=3  [3,0][3,1] ...       [3,7]

All accesses go through db_ref[0], and local indices are the original indices.
```

Index mapping:
- db_ref indices = [0]
- local indices = original indices

Transformation (Coarse):

```
  A : memref<[D0 x D1 x ...] x T>

        +------------------------------+
        | DbAlloc (mode=coarse)        |
        | outerSizes = [1]             |
        | innerSizes = [D0, D1, ...]   |
        +------------------------------+
                     |
                     v
        +------------------------------+
        | DbAcquire                    |
        | offsets=[0], sizes=[1]       |
        +------------------------------+
                     |
                     v
        +------------------------------+
        | DbRef[0] -> view             |
        | view : memref<[D0 x D1 ...]> |
        +------------------------------+
                     |
                     v
        memref.load/store view[...original indices...]
```

---

## 5) Mode 2 - Fine-Grained (Element-wise)

Meaning: one DB per element of the outer dimensions (outerRank > 0).

Theorem (Element-wise Layout):
If outerRank = r, then each unique outer index tuple selects a distinct
DB; remaining indices address inside that block.

Layout (2D example A[4][8], outerRank=1, partition dim 0):

```
outerSizes=[4]  (one DB per row)

  +-------+ +-------+ +-------+ +-------+
  | Row 0 | | Row 1 | | Row 2 | | Row 3 |
  +-------+ +-------+ +-------+ +-------+
   db_ref[0] db_ref[1] db_ref[2] db_ref[3]
```

Concrete example (2D, A[4][8], outerRank=1 on rows):

```
DB[0] (row 0) holds: [0,0] [0,1] ... [0,7]
DB[1] (row 1) holds: [1,0] [1,1] ... [1,7]
DB[2] (row 2) holds: [2,0] [2,1] ... [2,7]
DB[3] (row 3) holds: [3,0] [3,1] ... [3,7]

Access A[i,j]:
  db_ref index  = i
  local index   = j
```

Index mapping:
- db_ref index = outer indices (e.g., i)
- local indices = remaining indices (e.g., j)

Concrete example (2D, A[4][4], outerRank=2 on [i,j]):

```
outerSizes=[4,4]
innerSizes=[]  (each DB holds a single scalar)

DB grid (one DB per element):
  DB[0,0] holds A[0,0]   ...   DB[0,3] holds A[0,3]
  ...
  DB[3,0] holds A[3,0]   ...   DB[3,3] holds A[3,3]

Access A[i,j]:
  db_ref indices = [i, j]
  local indices  = []   (scalar)
```

Full-range acquire in fine-grained:
- allocation is still many DBs
- acquire offset = 0, size = numElements (all rows / tiles)
- used when access does not depend on the partition offset (or is indirect)

Transformation (Fine-Grained, outerRank=1 example):

```
  A : memref<[D0 x D1] x T>

        +-----------------------------------+
        | DbAlloc (mode=fine_grained)       |
        | partitionedDims = [0]             |
        | outerSizes = [D0]                 |
        | innerSizes = [D1]                 |
        +-----------------------------------+
                     |
                     v
        +-----------------------------------+
        | DbAcquire                          |
        | indices = [i]  (element coordinate)|
        +-----------------------------------+
                     |
                     v
        +-----------------------------------+
        | DbRef[i] -> view                  |
        | view : memref<[D1] x T> (row)     |
        +-----------------------------------+
                     |
                     v
        memref.load/store view[j]

  Full-range acquire variant (same allocation, different acquire):

        DbAcquire indices=[0] sizes=[D0]   (conceptual)
            |
            v
        DbRef[k] selects any row k (used for indirect / non-offset-tied access)
```

---

## 6) Mode 3 - Block (Blocked)

Meaning: each DB contains a contiguous block along one or more
dimensions.

Theorem (Block Layout):
If a dimension d is partitioned with block size B:
- block index = floor(global_d / B)
- local index = global_d mod B
db_ref chooses the block (relative to startBlock), local indices address inside
that block.

Layout (2D example A[4][8], blockSize=2 on dim 0):

```
outerSizes=[2]  (two chunks)
innerSizes=[2,8] (each DB holds 2 rows)

  +---------------+  +---------------+
  | Block 0       |  | Block 1       |
  | rows 0-1      |  | rows 2-3      |
  +---------------+  +---------------+
   db_ref[0]         db_ref[1]
```

Concrete example (2D, A[6][8], blockSize=2 on rows):

```
outerSizes=[3] blocks, each inner block is [2 x 8]

DB[0] holds rows 0..1:
  [0,0]...[0,7]
  [1,0]...[1,7]

DB[1] holds rows 2..3:
  [2,0]...[2,7]
  [3,0]...[3,7]

DB[2] holds rows 4..5:
  [4,0]...[4,7]
  [5,0]...[5,7]

Access A[i,j]:
  blockIdx   = floor(i / 2)
  localRow   = i % 2
  db_ref     = DB[blockIdx]
  local      = [localRow, j]
```

Index mapping (partition dim 0):
- db_ref index = i / B
- local indices = [i % B, j]

Concrete example (2D tiling, A[4][4], partitionedDims=[0,1], blockSizes=[2,2]):

```
outerSizes=[2,2] (db_ref indices order: [iBlock, jBlock])
innerSizes=[2,2]

Outer grid of DB tiles:

           j=0..1       j=2..3
        +-----------+-----------+
i=0..1  | DB[0,0]   | DB[0,1]   |
        | 2x2 tile  | 2x2 tile  |
        +-----------+-----------+
i=2..3  | DB[1,0]   | DB[1,1]   |
        | 2x2 tile  | 2x2 tile  |
        +-----------+-----------+

Example access A[i,j]:
  iBlock=floor(i/2), localI=i%2
  jBlock=floor(j/2), localJ=j%2
  db_ref indices = [iBlock, jBlock]
  local indices  = [localI, localJ]
```

Full-range acquire in block mode:
- allocation has many blocks
- acquire offset = 0, size = numBlocks
- used when access does not depend on the per-task partition offset (or is indirect)

Transformation (Block, 1D-blocked-on-rows example):

```
  A : memref<[D0 x D1] x T>
  blockSize = B  (partition dim 0)

        +--------------------------------------+
        | DbAlloc (mode=block)                 |
        | partitionedDims = [0]                |
        | outerSizes = [ceil(D0 / B)]          |
        | innerSizes = [B, D1]                 |
        +--------------------------------------+
                       |
                       v
        +--------------------------------------+
        | DbAcquire (per worker task)          |
        | offsets = [startBlock]               |
        | sizes   = [1]                        |
        +--------------------------------------+
                       |
                       v
        +--------------------------------------+
        | Index localization                   |
        | blockIdx  = floor(i / B)             |
        | localBlk  = blockIdx - startBlock    |
        | localRow  = i % B                    |
        +--------------------------------------+
                       |
                       v
        +--------------------------------------+
        | DbRef[localBlk] -> view              |
        | view : memref<[B x D1] x T>          |
        +--------------------------------------+
                       |
                       v
        memref.load/store view[localRow, j]

  Full-range acquire variant (same allocation, different acquire):

        DbAcquire offsets=[0], sizes=[numBlocks]
            |
            v
        DbRef[blockIdx] allows access to any block (used for indirect patterns)
```

---

## 7) Mode 4 - Stencil (Block + Halo / ESD)

Meaning: Block layout plus halos for neighbor access, implemented using ESD
(Ephemeral Slice Dependencies).

### 7.1 What ESD Is (Runtime Semantics)

ESD is a runtime capability: an EDT can depend on a *byte slice* of a DB.
When the DB becomes ready, the runtime signals the EDT with a pointer to:

  dbData + byteOffset

and a payload size. On multi-node, the runtime sends only that byte slice.

This is the right primitive for distributed ghost-cell exchange: minimal halo
traffic while preserving blocked compute.

### 7.2 Stencil Layout (Conceptual View)

Stencil keeps the same *allocation layout* as Block (outerSizes/innerSizes).
What changes is the *dependency delivery* to the EDT:

- owned: the worker's chunk (normal block acquire)
- leftHalo: a partial slice from the previous chunk (ESD)
- rightHalo: a partial slice from the next chunk (ESD)

Transformation (Stencil / ESD, row-partitioned example):

```
  A : memref<[D0 x D1] x T>
  blockSize=B, haloLeft=HL, haloRight=HR, partition dim 0

        +--------------------------------------+
        | DbAlloc (mode=stencil)               |
        | (same physical layout as Block)      |
        | outerSizes = [ceil(D0 / B)]          |
        | innerSizes = [B, D1]                 |
        +--------------------------------------+
                       |
                       v
        +--------------------------------------------------+
        | Per worker EDT dependencies (3-buffer delivery)  |
        +--------------------------------------------------+
           |                       |                       |
           v                       v                       v
  +-------------------+   +-------------------+   +-------------------+
  | owned acquire     |   | left halo acquire |   | right halo acquire|
  | offsets=[k],      |   | offsets=[k-1],    |   | offsets=[k+1],    |
  | sizes=[1]         |   | sizes=[1]         |   | sizes=[1]         |
  | full block        |   | ESD slice of DB   |   | ESD slice of DB   |
  +-------------------+   +-------------------+   +-------------------+
           |                       |                       |
           v                       v                       v
        owned view              leftHalo view           rightHalo view

  Multi-node note:
    halo acquires use ESD byteOffset/size, so the runtime transfers ONLY halo
    bytes between nodes (ghost-cell exchange), not the full neighbor block.
```

Example (1D rows, blockSize=4, haloLeft=1, haloRight=1, worker owns rows 4-7):

```
Global rows:
  ... | 3 | 4 5 6 7 | 8 | ...
        ^   owned    ^
     leftHalo      rightHalo

Delivered buffers (conceptual):
  leftHalo[0]  -> global row 3   (slice of previous chunk)
  owned[0..3]  -> global rows 4..7
  rightHalo[0] -> global row 8   (slice of next chunk)
```

Concrete example (2D, A[10][8], blockSize=4, haloLeft=1, haloRight=1):

Assume worker k=1 owns rows 4..7 (baseOffset=4, blockSize=4).

```
Row mapping (for any column j):
  A[3,j]  -> leftHalo[0, j]
  A[4,j]  -> owned[0, j]
  A[5,j]  -> owned[1, j]
  A[6,j]  -> owned[2, j]
  A[7,j]  -> owned[3, j]
  A[8,j]  -> rightHalo[0, j]

Correctness rule:
  stores are allowed only to owned[r,j]; halos are read-only inputs.
```

Index mapping (baseOffset semantics, partition dim is rows):

Let:
- baseOffset = global start row for owned chunk
- blockSize = number of owned rows
- haloLeft / haloRight = halo widths (rows)

Then for an access at globalRow:

- if globalRow < baseOffset:
  - use leftHalo
  - haloRow = (globalRow - baseOffset) + haloLeft
- else if globalRow < baseOffset + blockSize:
  - use owned
  - localRow = globalRow - baseOffset
- else:
  - use rightHalo
  - haloRow = globalRow - (baseOffset + blockSize)

Correctness rule:
- stores must target owned only; halos are read-only.

### 7.3 Plan: "ESD Per Row" Lowering (Avoid Per-Element Branching)

Problem:
- A naive stencil indexer that selects (owned vs halo) per load generates
  branch/select overhead in the innermost loop and blocks vectorization.

Goal:
- Pay halo logic per *row* (surface) rather than per *element* (volume).

Strategy (row-partitioned stencils like Jacobi2D):

1) Identify the loop nest where the partitioned dimension is the outer loop
   (row loop), and the inner loops iterate other dims (e.g., columns).
2) Split the row loop into three regions:
   - left boundary rows: [0, haloLeft)
   - interior rows: [haloLeft, blockSize - haloRight)
   - right boundary rows: [blockSize - haloRight, blockSize)
3) For the interior region, rewrite all stencil loads to use owned only:
   - no halo selection, no bounds checks
   - enables vectorization and reduces control overhead
4) For boundary regions, use specialized code paths that load the missing
   neighbor rows from the halo buffers.

For halo=1 (common 5-point / 7-point stencils), this reduces to:

```
interior rows: r = 1 .. blockSize-2   (owned-only)
first row:     r = 0                 (uses leftHalo for r-1)
last row:      r = blockSize-1        (uses rightHalo for r+1)
```

Theorem (Per-row ESD Overhead):
If halo widths are O(1) and blockSize is large, then:
- compute overhead from halo selection is O(haloLeft + haloRight) per chunk
- the innermost loop is branch-free in the interior

Per-row compute structure (conceptual, local row r in [0..B-1]):

```
  +--------------------+--------------------------------------------+
  | Region             | Neighbor row sources                        |
  +--------------------+--------------------------------------------+
  | left boundary      | (r-1) from leftHalo,  r and (r+1) from owned |
  | interior           | (r-1), r, (r+1) all from owned               |
  | right boundary     | (r+1) from rightHalo, r and (r-1) from owned |
  +--------------------+--------------------------------------------+
```

This preserves multi-node benefits (minimal halo transfers) and restores
single-node performance by avoiding per-element halo selection.

### 7.4 Chunk Alignment During Direct Materialization (Stencil Safety)

For stencil loops, worker chunking must stay consistent with DB block
boundaries, but it must not expand the logical iteration domain.

Current policy:

1) Explicit compile-time block hints (`PartitioningHint blockSize > 1`)
   - If loop lower bound is a known constant and misaligned, direct Core
     materialization may
     align the chunking base downward.
   - The generated inner loop is then clamped back to the original
     `[lowerBound, upperBound)` domain, so no extra iterations execute.

2) Runtime DB block-size hints (`runtimeBlockSizeHint`)
   - Use the runtime block size for chunk granularity.
   - Do not realign `lowerBound` at runtime.
   - This avoids domain expansion and tail/halo mismatches in stencil kernels.

Why runtime lower-bound alignment is disabled:
- With internode stencil chunking, expanding `lowerBound` (for example `1 -> 0`)
  can create acquire/iteration mismatches near the upper edge on tail chunks.
- In Jacobi-style loops (`i-1`, `i`, `i+1`), the final interior row requires
  stable ownership of the `i+1` neighbor row.

Example A (compile-time block alignment, safe):

```
Loop domain:      i in [5, 17), step=1
Explicit block:   B = 4
Aligned base:     4

Chunking domain:  [4, 17)   (for partition math)
Execution domain: [5, 17)   (after loopLower/loopUpper clamp)
```

Example B (runtime block hint, no runtime lower align):

```
Loop domain:        i in [1, 255), step=1     (254 interior rows)
numNodes = 6
runtime block hint: B = ceil(256 / 6) = 43

Chunk starts (global):
  node0: [1, 44)
  node1: [44, 87)
  node2: [87, 130)
  node3: [130, 173)
  node4: [173, 216)
  node5: [216, 255)

No runtime realignment to 0 is applied.
```

### 7.5 Stencil Limitations (Current)

- Stencil ESD is restricted to leading partitioned dimensions.
- If partitionedDims is non-leading, stencil is downgraded to Block.

---

## 8) Non-Leading Partitioning (partitionedDims)

CARTS supports partitioning non-leading dimensions for Block and Fine-grained.
The selected dimensions are recorded in partitionedDims and used by the block
indexer for correct localization.

Theorem (Non-Leading Block Layout):
Let partitionedDims = [p0, p1, ...] and blockSizes = [B0, B1, ...]. Then:
- outerSizes[k] = ceil(dimSize[p_k] / B_k)
- innerSizes keeps original dimension order, but replaces each partitioned dim
  size with its block size
- db_ref indices correspond to partitionedDims order
- local indices keep original order, with partitioned dims localized by modulo

Example A[i][j][k], partition only k (partitionedDims=[2], B=4):

```
outerSizes = [ceil(K/4)]
innerSizes = [I, J, 4]

db_ref index = k / 4
local indices = [i, j, k % 4]
```

Concrete example (3D, A[2][3][8], partitionedDims=[2], blockSize=4):

```
outerSizes=[2] (k in blocks of 4)
innerSizes=[2,3,4]

DB[0] contains k=0..3 (localK=0..3):
  A[0,0,0..3]  A[0,1,0..3]  A[0,2,0..3]
  A[1,0,0..3]  A[1,1,0..3]  A[1,2,0..3]

DB[1] contains k=4..7 (localK=0..3):
  A[0,0,4..7]  A[0,1,4..7]  A[0,2,4..7]
  A[1,0,4..7]  A[1,1,4..7]  A[1,2,4..7]

Access A[i,j,k]:
  blockIdx = floor(k / 4)
  localK   = k % 4
  db_ref   = DB[blockIdx]
  local    = [i, j, localK]
```

Transformation (Non-leading Block, partition k example):

```
  A : memref<[I x J x K] x T>
  partitionedDims=[2], blockSize=B

        +------------------------------------------+
        | DbAlloc (mode=block)                     |
        | partitionedDims = [2]                    |
        | outerSizes = [ceil(K / B)]               |
        | innerSizes = [I, J, B]                   |
        +------------------------------------------+
                       |
                       v
        +------------------------------------------+
        | DbAcquire (per worker task)              |
        | offsets = [startBlock]                   |
        | sizes   = [1]                            |
        +------------------------------------------+
                       |
                       v
        +------------------------------------------+
        | Index localization                        |
        | blockIdx = floor(k / B)                   |
        | localBlk = blockIdx - startBlock          |
        | localK   = k % B                          |
        +------------------------------------------+
                       |
                       v
        +------------------------------------------+
        | DbRef[localBlk] -> view                  |
        | view : memref<[I x J x B] x T>           |
        +------------------------------------------+
                       |
                       v
        memref.load/store view[i, j, localK]
```

Example A[i][j][k], partition i and k (partitionedDims=[0,2]):

```
outerSizes = [ceil(I/Bi), ceil(K/Bk)]
innerSizes = [Bi, J, Bk]

db_ref indices = [i / Bi, k / Bk]
local indices = [i % Bi, j, k % Bk]
```

Concrete example (3D, A[4][2][6], partitionedDims=[0,2], Bi=2, Bk=3):

```
outerSizes = [ceil(4/2)=2, ceil(6/3)=2]  (db_ref indices order: [iBlock, kBlock])
innerSizes = [2, 2, 3]

Outer grid of DBs (by ranges in original coordinates):

              k=0..2       k=3..5
           +-----------+-----------+
i=0..1     | DB[0,0]   | DB[0,1]   |
           +-----------+-----------+
i=2..3     | DB[1,0]   | DB[1,1]   |
           +-----------+-----------+

Example access A[i,j,k]:
  iBlock = floor(i / 2), localI = i % 2
  kBlock = floor(k / 3), localK = k % 3
  db_ref indices = [iBlock, kBlock]
  local indices  = [localI, j, localK]
```

---

## 9) Coarse vs Full-Range Summary (Quick Table)

| Concept | What changes | When used |
|--------|--------------|-----------|
| Coarse layout | outerSizes = [1]; single DB | No safe/profitable partitioning |
| Full-range acquire | acquire spans all blocks | Access is indirect or not tied to partition offset |

---

## 10) Where the Meaning Is Implemented

- Analysis (patterns, bounds, partition dims): DbAcquireNode / DbAllocNode
- Pass-facing acquire summary: `DbAnalysis::analyzeAcquirePartition`
- Decisions (H1.*): `evaluatePartitioningHeuristics` + H1.7 per-acquire voting
- Coordination + plan: DbPartitioning
- Mode reconciliation: local controller logic in `DbPartitioning`
- Block sizing + chosen dims: `DbBlockPlanResolver`
- Layout materialization: `CreateDbs` from SDE-authored attrs and
  `DbPhysicalLayoutPlan`
- Index localization for raw memref fallback: `DbBlockIndexer` /
  `DbElementWiseIndexer`
- ESD runtime semantics: byte-slice deps (`artsRecordDepAt`) + pointer signaling

This keeps semantics explicit: SDE pattern analysis and distribution planning
decide layout and task slices; SDE/Core materialization creates DB/acquire
objects; the remaining raw-memref indexers only localize compatibility bodies;
and the runtime implements the ESD transport.

---
