// RUN: %carts-compile %s --arts-config %arts_config --arts-epoch-finish-continuation --start-from=epochs --pipeline pre-lowering | %FileCheck %s
// RUN: %carts-compile %samples_dir/jacobi/for/jacobi-for.mlir --O3 --arts-config %arts_config --arts-epoch-finish-continuation --pipeline pre-lowering | %FileCheck %s --check-prefix=DEPSLOT

// Test that CPS chain param ordering attributes survive through pre-lowering.
// Verifies:
//   1. cps_chain_id is preserved on edt_create ops
//   2. cps_iter_counter_param_idx is present when the relaunch carries loop state
//   3. cps_outer_epoch_param_idx is present when the relaunch carries loop state
//   4. continuation_for_epoch marks EDTs as continuation-managed

// CHECK-DAG: arts.cps_chain_id = "{{.*chain_0}}", arts.cps_iter_counter_param_idx = {{[0-9]+}} : i64, arts.cps_outer_epoch_param_idx = {{[0-9]+}} : i64
// CHECK-DAG: arts.cps_chain_id = "{{.*chain_1}}"
// CHECK-NOT: scf.for

// A CPS continuation can rebuild multiple GUID tables from one compact
// dependency. Relaunching its sibling chain must place those rebuilt handles in
// the target continuation's ABI slots, not in local acquire walk order.
//
// DEPSLOT: func.func private @__arts_edt_{{[0-9]+}}{{.*}}arts.cps_param_perm = array<i64: 3, 4, 5, 6, 7, 8, 9>
// DEPSLOT: %[[FIRST_ALLOC:.+]] = memref.alloc() : memref<{{[0-9]+}}xi64>
// DEPSLOT: memref.store {{.*}}, %[[FIRST_ALLOC]][
// DEPSLOT: %[[SECOND_ALLOC:.+]] = memref.alloc() : memref<{{[0-9]+}}xi64>
// DEPSLOT: memref.store {{.*}}, %[[SECOND_ALLOC]][
// DEPSLOT: %[[FIRST_PTR:.+]] = polygeist.memref2pointer %[[FIRST_ALLOC]]
// DEPSLOT: %[[FIRST_RAW:.+]] = llvm.ptrtoint %[[FIRST_PTR]]
// DEPSLOT: %[[SECOND_PTR:.+]] = polygeist.memref2pointer %[[SECOND_ALLOC]]
// DEPSLOT: %[[SECOND_RAW:.+]] = llvm.ptrtoint %[[SECOND_PTR]]
// DEPSLOT: arts_rt.dep_db_acquire(%arg3) offset[%c4 : index]
// DEPSLOT: %[[SCRATCH_ALLOC:.+]] = memref.alloc() {{.*}} : memref<{{[0-9]+}}xi64>
// DEPSLOT: memref.store {{.*}}, %[[SCRATCH_ALLOC]][
// DEPSLOT: %[[SCRATCH_PTR:.+]] = polygeist.memref2pointer %[[SCRATCH_ALLOC]]
// DEPSLOT: %[[SCRATCH_RAW:.+]] = llvm.ptrtoint %[[SCRATCH_PTR]]
// DEPSLOT: arts_rt.edt_param_pack(%[[SCRATCH_RAW]], %[[FIRST_RAW]], %[[SECOND_RAW]],
// DEPSLOT: arts_rt.edt_create({{.*}}) {{.*}}arts.cps_param_perm = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {

  func.func private @work_a() -> ()
  func.func private @work_b() -> ()

  func.func @test_cps_relaunch_param_ordering() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c50 = arith.constant 50 : index
    %c0_i32 = arith.constant 0 : i32
    scf.for %t = %c0 to %c50 step %c1 {
      %e0 = arts.epoch {
        arts.edt <task> <intranode> route(%c0_i32) {
        ^bb0:
          func.call @work_a() : () -> ()
          arts.yield
        }
        arts.yield
      } : i64
      %e1 = arts.epoch {
        arts.edt <task> <intranode> route(%c0_i32) {
        ^bb0:
          func.call @work_b() : () -> ()
          arts.yield
        }
        arts.yield
      } : i64
    }
    return
  }
}
