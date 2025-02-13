//! Module instantiation.

use crate::lir::*;

/// Module Instantiation.
#[derive(Debug, Clone)]
pub struct ModuleInst {
    /// Input interface type.
    pub(crate) input_interface_typ: InterfaceTyp,
    /// Output interface type.
    pub(crate) output_interface_typ: InterfaceTyp,
    /// Module name.
    pub(crate) module_name: String,
    /// Instance name.
    pub(crate) inst_name: String,
    /// Parameters.
    pub(crate) params: Vec<(String, usize)>,
    /// Indicates that the module has the clock and reset signal.
    pub(crate) has_clkrst: bool,
    /// Input prefix.
    pub(crate) input_prefix: Option<String>,
    /// Output prefix.
    pub(crate) output_prefix: Option<String>,
    /// Shakeflow module.
    pub(crate) module: Option<Module>,
}

impl PrimitiveModule for ModuleInst {
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

impl ModuleInst {
    /// Generates module instantiation from a shakeflow module.
    pub fn from_module(
        module_name: String, inst_name: String, params: Vec<(String, usize)>, has_clkrst: bool,
        input_prefix: Option<String>, output_prefix: Option<String>, module: Module,
    ) -> Self {
        Self {
            input_interface_typ: module.inner.input_interface_typ(),
            output_interface_typ: module.inner.output_interface_typ(),
            module_name,
            inst_name,
            params,
            has_clkrst,
            input_prefix,
            output_prefix,
            module: Some(module),
        }
    }
}
