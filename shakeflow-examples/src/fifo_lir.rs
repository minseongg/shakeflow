//! FIFO implemented in LIR.

use shakeflow::*;
use shakeflow_std::*;

/// FIFO.
pub fn fifo(_elt_typ: lir::PortDecls, _capacity: usize, _pipe: bool, _flow: bool) {
    let module = lir::composite("fifo_lir", VrChannel::<u16>::interface_typ(), Some("in"), Some("out"), |value, k| {
        value.fsm(
            k,
            None,
            VrChannel::<u16>::interface_typ(),
            <()>::port_decls(),
            |i_fwd, o_bwd, s| {
                let o_fwd = i_fwd;
                let i_bwd = o_bwd;
                let s_next = s;
                (o_fwd, i_bwd, s_next)
            },
            lir::Expr::x(<()>::port_decls()).into(),
        )
    })
    .build();

    let module: vir::Module =
        codegen::gen_module::<Virgen>(String::from("fifo_lir"), &module).expect("Failed to codegen").into();

    println!("{}", module);
}
