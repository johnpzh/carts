// RUN: %carts-compile --print-pipeline-manifest-json | %FileCheck %s --check-prefix=MANIFEST

// Verifies production CODIR wiring: SDE planning no longer runs direct
// codelet lowering to ARTS, and executable `sde-planning` / `sde-to-codir` /
// `codir-to-arts` stages sit between frontend cleanup and ARTS refinement.

module {}

// MANIFEST: "sde-to-codir"
// MANIFEST: "codir-to-arts"
// MANIFEST: "name": "sde-planning"
// MANIFEST-NOT: "aliases"
// MANIFEST-NOT: "openmp-to-arts"
// MANIFEST-NOT: "ConvertSdeToArts"
// MANIFEST: "name": "sde-to-codir"
// MANIFEST: "ConvertSdeToCodir"
// MANIFEST: "CodirCodeletOpt"
// MANIFEST: "VerifyCodir"
// MANIFEST-SAME: "dependsOn": ["sde-planning"]
// MANIFEST: "name": "codir-to-arts"
// MANIFEST: "ConvertCodirToArts"
// MANIFEST-NOT: "ConvertSdeToArts"
// MANIFEST: "VerifySdeLowered"
// MANIFEST: "VerifyArtsObjectsOnly"
// MANIFEST-SAME: "dependsOn": ["sde-to-codir"]
// MANIFEST: "name": "post-db-refinement"
// MANIFEST: "PartialReductionSplitMaterialization"
// MANIFEST: "DistributedLaunchConsistency"
// MANIFEST-SAME: "dependsOn": ["create-dbs"]
