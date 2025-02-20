//! Low-level IR's module.

use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

use thiserror::Error;

use super::*;

/// Primitive modules.
pub trait PrimitiveModule: 'static + fmt::Debug {
    /// Returns module name.
    fn get_module_name(&self) -> String;

    /// Returns input interface.
    fn input_interface_typ(&self) -> InterfaceTyp;

    /// Returns output interface.
    fn output_interface_typ(&self) -> InterfaceTyp;
}

/// Module's inner data.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ModuleInner {
    /// Composite module comprising submodules.
    Composite(String, CompositeModule),

    /// FSM.
    Fsm(Fsm),

    /// Module instantiation.
    ModuleInst(ModuleInst),

    /// Virtual module
    VirtualModule(VirtualModule),
}

/// Module.
#[derive(Debug, Clone)]
pub struct Module {
    /// Inner.
    pub inner: Rc<ModuleInner>,
}

impl Module {
    /// Split a module.
    ///
    /// NOTE: This should be applied for virtual module.
    pub fn split(self) -> (Module, Module) {
        match self.inner.deref() {
            ModuleInner::VirtualModule(virtual_module) => {
                let (in1, in2) = match virtual_module.input_interface_typ() {
                    InterfaceTyp::Struct(fields) => {
                        assert_eq!(fields.len(), 2);
                        let mut iter = fields.into_iter();
                        (iter.next().unwrap(), iter.next().unwrap())
                    }
                    _ => todo!(),
                };
                let (out1, out2) = match virtual_module.output_interface_typ() {
                    InterfaceTyp::Struct(fields) => {
                        assert_eq!(fields.len(), 2);
                        let mut iter = fields.into_iter();
                        (iter.next().unwrap(), iter.next().unwrap())
                    }
                    _ => todo!(),
                };
                let vm1 = VirtualModule {
                    registered_index: virtual_module.registered_index,
                    module_name: virtual_module.module_name.clone(),
                    input_prefix: virtual_module.input_prefix.clone(),
                    output_prefix: virtual_module.output_prefix.clone(),
                    input_interface_typ: virtual_module.input_interface_typ(),
                    output_interface_typ: virtual_module.output_interface_typ(),
                    input_endpoint_path: std::iter::once(EndpointNode::Field(in1.0))
                        .chain(virtual_module.input_endpoint().inner)
                        .collect(),
                    output_endpoint_path: std::iter::once(EndpointNode::Field(out1.0))
                        .chain(virtual_module.output_endpoint().inner)
                        .collect(),
                };
                let vm2 = VirtualModule {
                    registered_index: virtual_module.registered_index,
                    module_name: virtual_module.module_name.clone(),
                    input_prefix: virtual_module.input_prefix.clone(),
                    output_prefix: virtual_module.output_prefix.clone(),
                    input_interface_typ: virtual_module.input_interface_typ(),
                    output_interface_typ: virtual_module.output_interface_typ(),
                    input_endpoint_path: std::iter::once(EndpointNode::Field(in2.0))
                        .chain(virtual_module.input_endpoint().inner)
                        .collect(),
                    output_endpoint_path: std::iter::once(EndpointNode::Field(out2.0))
                        .chain(virtual_module.output_endpoint().inner)
                        .collect(),
                };
                (vm1.into(), vm2.into())
            }
            _ => panic!("internal compiler error: split api can only be used for Virtual Modules"),
        }
    }

    /// Returns module name.
    pub fn get_module_name(&self) -> String {
        match &*self.inner {
            ModuleInner::Composite(name, _) => name.clone(),
            ModuleInner::Fsm(module) => module.get_module_name(),
            ModuleInner::ModuleInst(module) => module.get_module_name(),
            ModuleInner::VirtualModule(module) => module.get_module_name(),
        }
    }

    /// Scan submodule instantiation for current module
    // TODO: Cycle detection
    pub fn scan_submodule_inst(&self) -> Vec<Module> {
        // scan for registered modules
        match &*self.inner {
            ModuleInner::Composite(_, composite_module) => composite_module.scan_submodule_inst(),
            ModuleInner::Fsm(_) | ModuleInner::VirtualModule(_) => vec![],
            ModuleInner::ModuleInst(module_inst) => {
                if let Some(module) = &module_inst.module {
                    [module.scan_submodule_inst(), vec![module.clone()]].concat()
                } else {
                    vec![]
                }
            }
        }
    }

    /// Walk the module structure and return a vec of mutable refs to names of all inner `ModuleInst`s.
    // TODO: Cycle detection
    pub fn scan_module_inst(&mut self) -> Vec<&mut ModuleInst> {
        // scan for registered modules
        match Rc::get_mut(&mut self.inner).unwrap() {
            ModuleInner::Composite(_, composite_module) => composite_module.scan_module_inst(),
            ModuleInner::Fsm(_) | ModuleInner::VirtualModule(_) => vec![],
            ModuleInner::ModuleInst(module_inst) => {
                vec![module_inst]
            }
        }
    }

    pub(crate) fn is_interface_eq(&self, other: Self) -> bool {
        (self.inner.input_interface_typ() == other.inner.input_interface_typ())
            && (self.inner.output_interface_typ() == other.inner.output_interface_typ())
            && (self.inner.input_prefix() == other.inner.input_prefix())
            && (self.inner.output_prefix() == other.inner.output_prefix())
    }
}

#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ModuleError {
    #[error("file error")]
    FileError(#[from] std::io::Error),
    #[error("the types are mismatched: {0}")]
    TypMismatch(String),
    #[error("misc error: {0}")]
    Misc(String),
}

impl ModuleInner {
    /// Returns input interface type of the module.
    pub fn input_interface_typ(&self) -> InterfaceTyp {
        match self {
            Self::Composite(_, builder) => builder.input_interface_typ(),
            Self::Fsm(module) => module.input_interface_typ(),
            Self::ModuleInst(module) => module.input_interface_typ(),
            ModuleInner::VirtualModule(module) => module.input_interface_typ(),
        }
    }

    /// Returns input prefix of the module.
    pub fn input_prefix(&self) -> Option<String> {
        match self {
            Self::Composite(_, builder) => builder.input_prefix.clone(),
            Self::Fsm(_) | Self::ModuleInst(_) | Self::VirtualModule(_) => None,
        }
    }

    /// Returns output interface type of the module.
    pub fn output_interface_typ(&self) -> InterfaceTyp {
        match self {
            Self::Composite(_, builder) => builder.output_interface_typ(),
            Self::Fsm(module) => module.output_interface_typ(),
            Self::ModuleInst(module) => module.output_interface_typ(),
            Self::VirtualModule(module) => module.output_interface_typ(),
        }
    }

    /// Returns output prefix of the module.
    pub fn output_prefix(&self) -> Option<String> {
        match self {
            Self::Composite(_, builder) => builder.output_prefix.clone(),
            Self::Fsm(_) | Self::ModuleInst(_) | Self::VirtualModule(_) => None,
        }
    }
}

impl From<Fsm> for Module {
    fn from(module: Fsm) -> Module {
        Module { inner: Rc::new(ModuleInner::Fsm(module)) }
    }
}

impl From<ModuleInst> for Module {
    fn from(module: ModuleInst) -> Module {
        Module { inner: Rc::new(ModuleInner::ModuleInst(module)) }
    }
}

impl From<VirtualModule> for Module {
    fn from(module: VirtualModule) -> Module {
        Module { inner: Rc::new(ModuleInner::VirtualModule(module)) }
    }
}
