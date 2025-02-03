//! Finite state machine (Mealy machine).

use std::fmt;
use std::marker::PhantomData;
use std::ops::*;

use crate::hir::*;
use crate::*;

/// Finite state machine (Mealy machine).
#[derive(Clone)]
pub struct Fsm<
    I: Interface,
    O: Interface,
    S: Signal,
    F: Fn(Expr<I::Fwd>, Expr<O::Bwd>, Expr<S>) -> (Expr<O::Fwd>, Expr<I::Bwd>, Expr<S>),
> {
    /// Module name.
    module_name: String,
    /// FSM function.
    pub(crate) f: F,
    /// Initial value of registers in the FSM.
    pub(crate) init: Expr<S>,
    _marker: PhantomData<(I, O)>,
}

impl<
        I: Interface,
        O: Interface,
        S: Signal,
        F: Fn(Expr<I::Fwd>, Expr<O::Bwd>, Expr<S>) -> (Expr<O::Fwd>, Expr<I::Bwd>, Expr<S>),
    > Fsm<I, O, S, F>
{
    /// Creates a new FSM.
    pub fn new(module_name: &str, f: F, init: Expr<S>) -> Self {
        Self { module_name: module_name.to_string(), f, init, _marker: PhantomData }
    }
}

impl<
        I: Interface,
        O: Interface,
        S: Signal,
        F: Fn(Expr<I::Fwd>, Expr<O::Bwd>, Expr<S>) -> (Expr<O::Fwd>, Expr<I::Bwd>, Expr<S>),
    > fmt::Debug for Fsm<I, O, S, F>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fsm").field("init", &self.init).finish()
    }
}

impl<
        I: Interface,
        O: Interface,
        S: Signal,
        F: 'static + Fn(Expr<I::Fwd>, Expr<O::Bwd>, Expr<S>) -> (Expr<O::Fwd>, Expr<I::Bwd>, Expr<S>),
    > From<Fsm<I, O, S, F>> for lir::Fsm
{
    fn from(module: Fsm<I, O, S, F>) -> Self {
        lir::Fsm::new(
            module.module_name,
            I::interface_typ(),
            O::interface_typ(),
            I::Fwd::port_decls(),
            O::Bwd::port_decls(),
            S::port_decls(),
            |i_fwd: lir::ExprId, o_bwd: lir::ExprId, s: lir::ExprId| {
                let (i_fwd, o_bwd, s) = (Expr::from(i_fwd), Expr::from(o_bwd), Expr::from(s));
                let (o_fwd, i_bwd, s) = (module.f)(i_fwd, o_bwd, s);
                (o_fwd.into_inner(), i_bwd.into_inner(), s.into_inner())
            },
            module.init.into_inner(),
        )
    }
}
