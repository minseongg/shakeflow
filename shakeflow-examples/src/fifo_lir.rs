//! FIFO implemented in LIR.

use shakeflow::lir::{ChannelTyp, InterfaceTyp, Shape};
use shakeflow::*;

/// Returns FIFO IO type.
pub fn fifo_io_typ(elt: lir::PortDecls) -> InterfaceTyp {
    let fwd =
        lir::PortDecls::Struct(vec![(None, elt), (Some("valid".to_string()), lir::PortDecls::Bits(Shape::new([1])))]);
    let bwd = lir::PortDecls::Struct(vec![(Some("ready".to_string()), lir::PortDecls::Bits(Shape::new([1])))]);
    InterfaceTyp::Channel(ChannelTyp { fwd, bwd })
}

/// Returns FIFO state type.
pub fn fifo_s_typ(elt: lir::PortDecls, capacity: usize) -> lir::PortDecls {
    lir::PortDecls::Struct(vec![
        (Some("inner".to_string()), elt.multiple(capacity)),
        (Some("raddr".to_string()), lir::PortDecls::Bits(Shape::new([clog2(capacity)]))),
        (Some("waddr".to_string()), lir::PortDecls::Bits(Shape::new([clog2(capacity)]))),
        (Some("len".to_string()), lir::PortDecls::Bits(Shape::new([clog2(capacity + 1)]))),
    ])
}

/// Returns FIFO state init value.
pub fn fifo_s_init(elt: lir::PortDecls, capacity: usize) -> lir::ExprId {
    lir::Expr::unproj(vec![
        (Some("inner".to_string()), lir::Expr::x(elt)),
        (Some("raddr".to_string()), lir::Expr::from_usize(0, clog2(capacity))),
        (Some("waddr".to_string()), lir::Expr::from_usize(0, clog2(capacity))),
        (Some("len".to_string()), lir::Expr::from_usize(0, clog2(capacity + 1))),
    ])
}

/// FIFO.
// TODO: Add `pipe` and `flow` option.
pub fn fifo(elt_typ: lir::PortDecls, capacity: usize) {
    let module = lir::composite("fifo_lir", fifo_io_typ(elt_typ.clone()), Some("in"), Some("out"), |value, k| {
        value.fsm(
            k,
            None,
            fifo_io_typ(elt_typ.clone()),
            fifo_s_typ(elt_typ.clone(), capacity),
            |ip, er, s| {
                let ip_inner = lir::Expr::member(ip, 0);
                let ip_valid = lir::Expr::member(ip, 1);

                let er_ready = lir::Expr::member(er, 0);

                let inner = lir::Expr::member(s, 0);
                let raddr = lir::Expr::member(s, 1);
                let waddr = lir::Expr::member(s, 2);
                let len = lir::Expr::member(s, 3);

                let empty = lir::Expr::is_eq(len, lir::Expr::from_usize(0, clog2(capacity + 1)));
                let full = lir::Expr::is_eq(len, lir::Expr::from_usize(capacity, clog2(capacity + 1)));

                let enq = lir::Expr::bitand(ip_valid, lir::Expr::not(full));
                let _deq = lir::Expr::bitand(er_ready, lir::Expr::not(empty));

                let ep = lir::Expr::unproj(vec![
                    (None, lir::Expr::get(elt_typ.clone(), inner, raddr)),
                    (Some("valid".to_string()), lir::Expr::not(empty)),
                ]);
                let ir = lir::Expr::unproj(vec![(Some("ready".to_string()), lir::Expr::not(full))]);

                // TODO: Change next state calculation logic as follows:
                //
                // ```
                // let inner_next = if enq { inner.set(waddr, ip.unwrap()) } else { inner };
                // let len_next = (len + U::from(enq).resize() - if deq { pop.resize() } else { 0.into_u() }).resize();
                // let raddr_next = if deq { wrapping_add::<{ clog2(N) }>(raddr, pop.resize(), N.into_u()) } else { raddr };
                // let waddr_next = if enq { wrapping_inc::<{ clog2(N) }>(waddr, N.into_u()) } else { waddr };
                // ```

                let inner_next = lir::Expr::cond(enq, lir::Expr::set(inner, waddr, ip_inner), inner);
                let len_next = len;
                let raddr_next = raddr;
                let waddr_next = waddr;

                let s_next = lir::Expr::unproj(vec![
                    (Some("inner".to_string()), inner_next),
                    (Some("raddr".to_string()), raddr_next),
                    (Some("waddr".to_string()), waddr_next),
                    (Some("len".to_string()), len_next),
                ]);

                (ep, ir, s_next)
            },
            fifo_s_init(elt_typ.clone(), capacity),
        )
    })
    .build();

    let module: vir::Module =
        codegen::gen_module::<Virgen>(String::from("fifo_lir"), &module).expect("Failed to codegen").into();

    println!("{}", module);
}
