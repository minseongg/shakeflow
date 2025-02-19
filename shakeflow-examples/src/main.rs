#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod arbiter;
mod custom_channel;
mod enum_signal;
mod fifo_lir;
mod fir_filter;
mod int_adder;
mod pulse_merge;
mod set;
mod split;
mod spsc_lir;
mod tree_fold;
mod virtual_module;

use std::path::Path;

use shakeflow::{Package, PackageError};

fn main() -> Result<(), PackageError> {
    {
        use shakeflow::*;
        use shakeflow_std::VrChannel;
        fifo_lir::fifo(<u16>::port_decls(), 8);
        spsc_lir::channel(VrChannel::<u16>::interface_typ());
    };

    let mut package = Package::default();
    package.add(arbiter::arbiter());
    package.add(fir_filter::fir_filter::<4, 3>([[false, false, true, false], [true, false, true, true], [
        true, true, true, false,
    ]]));
    package.add(int_adder::int_adder());
    package.add(pulse_merge::pulse_merge());
    package.add(set::m::<10>());
    package.add(split::m::<32, 80>("split_test"));
    package.add(split::m::<512, 112>("split_test_small_header"));
    package.add(tree_fold::m());
    package.add(enum_signal::m());
    package.add(virtual_module::feedback_test_m());
    package.add(virtual_module::read_write_test_m());
    package.add(virtual_module::read_write_array_test_m());
    package.add(virtual_module::read_write_inline_test_m());
    package.add(virtual_module::spsc_test_m());
    package.add(::shakeflow_std::cuckoo_table::m());
    package.gen_vir(Path::new("./build"))
}
