//! FIFO implemented in LIR.

use shakeflow::codegen::{Codegen, Context};
use shakeflow::*;
use shakeflow_std::*;

/// FIFO.
pub fn fifo(_elt_typ: lir::PortDecls, _capacity: usize, _pipe: bool, _flow: bool) {
    let fsm = lir::Fsm::new(
        "fifo".to_string(),
        VrChannel::<bool>::interface_typ(),
        VrChannel::<bool>::interface_typ(),
        Valid::<bool>::port_decls(),
        Ready::port_decls(),
        <()>::port_decls(),
        |i_fwd, o_bwd, s| {
            let o_fwd = i_fwd;
            let i_bwd = o_bwd;
            let s_next = s;
            (o_fwd, i_bwd, s_next)
        },
        lir::Expr::x(<()>::port_decls()).into(),
    );

    let virgen = Virgen;
    let mut ctx = Context::new();
    ctx.enter_scope("fifo".to_string());

    let _module_items = virgen.gen_module_fsm(&fsm, &mut ctx).expect("Failed to generate module items");

    // println!("{:#?}", module_items);
}
