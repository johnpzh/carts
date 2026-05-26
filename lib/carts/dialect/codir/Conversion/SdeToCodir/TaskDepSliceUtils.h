#ifndef CARTS_DIALECT_CODIR_CONVERSION_SDETOCODIR_TASKDEPSLICEUTILS_H
#define CARTS_DIALECT_CODIR_CONVERSION_SDETOCODIR_TASKDEPSLICEUTILS_H

#include "carts/dialect/sde/IR/SdeDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "polygeist/Ops.h"

namespace mlir::carts::codir {

bool hasCompleteMuDepSlice(sde::SdeMuDepOp muDep);
bool hasOnlyStaticMuDepSliceBounds(sde::SdeMuDepOp muDep);
bool hasCodirTaskDepSliceBoundsSupport(sde::SdeMuDepOp muDep);

bool subviewMatchesMuDepSlice(memref::SubViewOp subview, sde::SdeMuDepOp muDep);
bool subindexMatchesMuDepSlice(polygeist::SubIndexOp subindex,
                               sde::SdeMuDepOp muDep);
bool haveSameMuDepSlice(sde::SdeMuDepOp lhs, sde::SdeMuDepOp rhs);
bool accessIndicesMatchMuDepOffsets(ValueRange indices, sde::SdeMuDepOp muDep);
bool rootAccessMatchesMuDepOffsets(Operation *user, sde::SdeMuDepOp muDep);

bool hasOnlyDirectLoadStoreUsersInTask(Value memref, Region &taskRegion);
bool hasExactSubviewAccessProofInTask(sde::SdeMuDepOp muDep,
                                      Region &taskRegion);
bool hasExactRootAccessProofInTask(sde::SdeMuDepOp muDep, Region &taskRegion);
bool hasPartitionedExactAccessProofInTask(Value source,
                                          ArrayRef<sde::SdeMuDepOp> sourceDeps,
                                          Region &taskRegion);

void collectExactSubviewAccessProofsInTask(
    sde::SdeMuDepOp muDep, Region &taskRegion,
    SmallVectorImpl<memref::SubViewOp> &subviews);
void collectExactSubindexAccessProofsInTask(
    sde::SdeMuDepOp muDep, Region &taskRegion,
    SmallVectorImpl<polygeist::SubIndexOp> &subindices);
void collectExactRootAccessProofsInTask(sde::SdeMuDepOp muDep,
                                        Region &taskRegion,
                                        SmallVectorImpl<Operation *> &accesses);

} // namespace mlir::carts::codir

#endif // CARTS_DIALECT_CODIR_CONVERSION_SDETOCODIR_TASKDEPSLICEUTILS_H
