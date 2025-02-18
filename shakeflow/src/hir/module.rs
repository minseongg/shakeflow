//! Module.

use std::marker::PhantomData;
use std::ops::*;

use crate::hir::*;
use crate::*;

/// Module.
#[derive(Debug)]
pub struct Module<I: Interface, O: Interface> {
    #[allow(missing_docs)]
    pub inner: lir::Module,
    _marker: PhantomData<(I, O)>,
}

impl<I: Interface, O: Interface> Module<I, O> {
    /// Creates new module.
    pub fn new(inner: lir::Module) -> Self {
        Module { inner, _marker: PhantomData }
    }
}

impl<I1: Interface, I2: Interface, O1: Interface, O2: Interface> Module<(I1, I2), (O1, O2)> {
    /// Split a module
    ///
    /// TODO: generalized to N-M modules?
    pub fn split(self) -> (Module<I1, O1>, Module<I2, O2>) {
        let (vm1, vm2) = self.inner.split();
        (Module::new(vm1), Module::new(vm2))
    }
}

impl<
        I: Interface,
        O: Interface,
        S: Signal,
        F: 'static + Fn(Expr<I::Fwd>, Expr<O::Bwd>, Expr<S>) -> (Expr<O::Fwd>, Expr<I::Bwd>, Expr<S>),
    > From<Fsm<I, O, S, F>> for Module<I, O>
{
    fn from(module: Fsm<I, O, S, F>) -> Self {
        Self { inner: lir::Fsm::from(module).into(), _marker: PhantomData }
    }
}

impl<I: Interface, O: Interface> From<ModuleInst<I, O>> for Module<I, O> {
    fn from(module: ModuleInst<I, O>) -> Self {
        Self { inner: lir::ModuleInst::from(module).into(), _marker: PhantomData }
    }
}

impl<I: Interface, O: Interface> Clone for Module<I, O> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), _marker: PhantomData }
    }
}
