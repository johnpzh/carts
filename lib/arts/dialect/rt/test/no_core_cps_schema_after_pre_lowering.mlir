// RUN: %carts-compile %s --arts-config %arts_config --arts-epoch-finish-continuation --start-from=epochs --pipeline pre-lowering | %FileCheck %s --check-prefix=EPOCH --implicit-check-not={{arts[.]c}}ps_ --implicit-check-not={{c}}ps_param_perm --implicit-check-not={{c}}ps_dep_routing
// RUN: %carts-compile %samples_dir/jacobi/for/jacobi-for.mlir --O3 --arts-config %arts_config --arts-epoch-finish-continuation --pipeline pre-lowering | %FileCheck %s --check-prefix=JACOBI --implicit-check-not={{arts[.]c}}ps_ --implicit-check-not={{c}}ps_param_perm --implicit-check-not={{c}}ps_dep_routing

// Repeated epoch loops lower through ordinary runtime epoch/EDT operations.

// EPOCH-LABEL: func.func @test_epoch_loop_pre_lowering
// EPOCH: scf.for
// EPOCH: arts_rt.create_epoch
// EPOCH: arts_rt.edt_param_pack()
// EPOCH: arts_rt.edt_create
// EPOCH: arts_rt.wait_on_epoch

// JACOBI: arts_rt.edt_create
// JACOBI: arts_rt.dep_bind

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func private @work_a() -> ()
  func.func private @work_b() -> ()

  func.func @test_epoch_loop_pre_lowering() {
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
