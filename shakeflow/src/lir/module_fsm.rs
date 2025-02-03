//! Finite state machine (Mealy machine).

use crate::lir::*;

/// Finite state machine (Mealy machine).
#[derive(Debug)]
pub struct Fsm {
    /// Input interface type.
    pub(crate) input_interface_typ: InterfaceTyp,
    /// Output interface type.
    pub(crate) output_interface_typ: InterfaceTyp,
    /// Module name.
    pub(crate) module_name: String,
    /// Output foreward expr.
    pub(crate) output_fwd: ExprId,
    /// Input backward expr.
    pub(crate) input_bwd: ExprId,
    /// State.
    pub(crate) state: ExprId,
    /// Initial value of registers in the FSM.
    pub(crate) init: ExprId,
}

impl Fsm {
    /// Creates a new fsm.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        module_name: String, input_interface_typ: InterfaceTyp, output_interface_typ: InterfaceTyp, s_typ: PortDecls,
        f: impl FnOnce(ExprId, ExprId, ExprId) -> (ExprId, ExprId, ExprId), init: ExprId,
    ) -> Self {
        let i_fwd_typ = input_interface_typ.fwd();
        let o_bwd_typ = output_interface_typ.bwd();

        let i_fwd = Expr::input(i_fwd_typ, Some("in".to_string())).into();
        let o_bwd = Expr::input(o_bwd_typ, Some("out".to_string())).into();
        let s = Expr::input(s_typ, Some("st".to_string())).into();
        let (o_fwd, i_bwd, s) = f(i_fwd, o_bwd, s);

        Fsm {
            input_interface_typ,
            output_interface_typ,
            module_name,
            output_fwd: o_fwd,
            input_bwd: i_bwd,
            state: s,
            init,
        }
    }
}

impl PrimitiveModule for Fsm {
    #[inline]
    fn get_module_name(&self) -> String {
        self.module_name.clone()
    }

    #[inline]
    fn input_interface_typ(&self) -> InterfaceTyp {
        self.input_interface_typ.clone()
    }

    #[inline]
    fn output_interface_typ(&self) -> InterfaceTyp {
        self.output_interface_typ.clone()
    }
}
