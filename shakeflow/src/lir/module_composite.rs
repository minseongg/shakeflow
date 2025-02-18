//! Composite module.

use std::collections::HashMap;
use std::mem;
use std::rc::Rc;

use thiserror::Error;

use super::*;

#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum CompositeModuleError {
    #[error("there are no submodules at the specified index")]
    NoSubmodules,
    #[error("there are no wires at the specified index")]
    NoWires,
    #[error("there are no channels at the specified index")]
    NoChannels,
    #[error("the specified endpoint is already occupied")]
    EndpointOccupied,
    #[error("the types are mismatched")]
    TypMismatch,
}

/// Composite module type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeModuleTyp {
    /// `I` -> `O`
    OneToOne,

    /// `[I; N]` -> `[O; N]`
    NToN(usize),
}

impl Default for CompositeModuleTyp {
    fn default() -> Self {
        Self::OneToOne
    }
}

/// Composite module.
#[derive(Debug, Default, Clone)]
pub struct CompositeModule {
    /// Name of module.
    pub name: String,

    /// Type of module.
    pub module_typ: CompositeModuleTyp,

    /// Inner submodules.
    pub submodules: Vec<(Module, Interface)>,

    /// Registered modules
    pub registered_modules: Vec<Module>,

    /// Input interface.
    pub input_interface: Interface,

    /// Input interface's prefix. For example, 's_axis' is input prefix for cmac_pad.
    pub input_prefix: Option<String>,

    /// Output interface.
    pub output_interface: Interface,

    /// Output interface's prefix. For example, 'm_axis' is output prefix for cmac_pad.
    pub output_prefix: Option<String>,
}

impl CompositeModule {
    /// Creates a new composite module with given prefix for input and output channels.
    pub fn new(name: String, input_prefix: Option<String>, output_prefix: Option<String>) -> Self {
        Self {
            name,
            module_typ: CompositeModuleTyp::default(),
            submodules: Vec::default(),
            registered_modules: Vec::default(),
            input_interface: Interface::default(),
            input_prefix,
            output_interface: Interface::default(),
            output_prefix,
        }
    }

    /// "register inline".
    pub fn register_inline(&mut self, module: Module) -> Module {
        assert!(
            matches!(&*module.inner, ModuleInner::Composite(..)),
            "Only `CompositeModule` can be registered to the context"
        );
        let registered_index = self.registered_modules.len();
        self.registered_modules.push(module.clone());
        let virtual_module = VirtualModule {
            module_name: module.get_module_name(),
            registered_index,
            input_prefix: module.inner.input_prefix().unwrap_or_else(|| "in".to_string()),
            output_prefix: module.inner.output_prefix().unwrap_or_else(|| "out".to_string()),
            input_interface_typ: module.inner.input_interface_typ(),
            input_endpoint_path: EndpointPath::default(),
            output_interface_typ: module.inner.output_interface_typ(),
            output_endpoint_path: EndpointPath::default(),
        };
        virtual_module.into()
    }

    /// Returns input interface type of the module.
    pub fn input_interface_typ(&self) -> InterfaceTyp {
        match self.module_typ {
            CompositeModuleTyp::OneToOne => self.input_interface.typ(),
            CompositeModuleTyp::NToN(n) => InterfaceTyp::Array(Box::new(self.input_interface.typ()), n),
        }
    }

    /// Returns output interface type of the module.
    pub fn output_interface_typ(&self) -> InterfaceTyp {
        match self.module_typ {
            CompositeModuleTyp::OneToOne => self.output_interface.typ(),
            CompositeModuleTyp::NToN(n) => InterfaceTyp::Array(Box::new(self.output_interface.typ()), n),
        }
    }

    /// Adds a submodule.
    pub fn add_submodule(&mut self, module: Module, input_interface: Interface) -> Interface {
        // Inserts the given module.
        let index = self.submodules.len();
        self.submodules.push((module.clone(), input_interface));

        // Calculates the output interface.
        module
            .inner
            .output_interface_typ()
            .into_primitives()
            .into_iter()
            .map(|(primitive_typ, path)| {
                (
                    match primitive_typ {
                        InterfaceTyp::Unit => Interface::Unit,
                        InterfaceTyp::Channel(channel_typ) => Interface::Channel(Channel {
                            typ: channel_typ,
                            endpoint: Endpoint::submodule(index, path.clone()),
                        }),
                        _ => panic!("not primitive type"),
                    },
                    path,
                )
            })
            .collect()
    }

    /// Builds a new module.
    pub fn build(self) -> Module {
        let name = self.name.clone();
        Module { inner: Rc::new(ModuleInner::Composite(name, self)) }
    }

    /// Builds a new module for array interface.
    pub fn build_array(mut self, name: &str, n: usize) -> Module {
        self.module_typ = CompositeModuleTyp::NToN(n);

        Module { inner: Rc::new(ModuleInner::Composite(String::from(name), self)) }
    }

    /// Scan submodule instantiation of composite module
    pub fn scan_submodule_inst(&self) -> Vec<Module> {
        ::std::iter::empty()
            .chain(self.submodules.iter().map(|(module, _)| module))
            .chain(self.registered_modules.iter())
            .flat_map(|module| module.scan_submodule_inst())
            .collect()
    }

    /// Walk the module structure and return a vec of mutable refs to names of all inner `ModuleInst`s.
    pub fn scan_module_inst(&mut self) -> Vec<&mut ModuleInst> {
        ::std::iter::empty()
            .chain(self.submodules.iter_mut().map(|(module, _)| module))
            .chain(self.registered_modules.iter_mut())
            .flat_map(|module| module.scan_module_inst())
            .collect()
    }
}

/// Creates a new composite module with given prefix for input and output channels.
pub fn composite(
    name: &str, interface_typ: InterfaceTyp, input_prefix: Option<&str>, output_prefix: Option<&str>,
    f: impl FnOnce(Interface, &mut CompositeModule) -> Interface,
) -> CompositeModule {
    CompositeModule::new(name.to_string(), input_prefix.map(String::from), output_prefix.map(String::from))
        .wrap(interface_typ, |_, iw, o| (o, iw))
        .and_then(f)
}

/// Creates new input interface from given interface type.
pub fn input_interface(typ: &InterfaceTyp) -> Interface {
    typ.into_primitives()
        .into_iter()
        .map(|(typ, path)| {
            (
                match typ {
                    InterfaceTyp::Unit => Interface::Unit,
                    InterfaceTyp::Channel(channel_typ) => {
                        Interface::Channel(Channel { typ: channel_typ, endpoint: Endpoint::input(path.clone()) })
                    }
                    _ => panic!("not primitive type"),
                },
                path,
            )
        })
        .collect()
}

/// Creates new temporary interface from given interface type.
pub fn temp_interface(typ: &InterfaceTyp) -> Interface {
    typ.into_primitives()
        .into_iter()
        .map(|(typ, path)| {
            (
                match typ {
                    InterfaceTyp::Unit => Interface::Unit,
                    InterfaceTyp::Channel(channel_typ) => {
                        Interface::Channel(Channel { typ: channel_typ, endpoint: Endpoint::temp(path.clone()) })
                    }
                    _ => panic!("not primitive type"),
                },
                path,
            )
        })
        .collect()
}

impl CompositeModule {
    /// `wrap` in LIR.
    ///
    /// For more details, please consult `wrap` method in HIR.
    pub fn wrap(
        mut self, iw_interface_typ: InterfaceTyp,
        f: impl FnOnce(&mut CompositeModule, Interface, Interface) -> (Interface, Interface),
    ) -> CompositeModule {
        // Takes old input/output interface.
        let old_output_interface = mem::take(&mut self.output_interface);
        let old_submodules_len = self.submodules.len();

        // Creates old input interface and new output interface.
        let (old_input_interface, new_output_interface) = {
            let new_input_interface = input_interface(&iw_interface_typ);
            let output_interface = temp_interface(&old_output_interface.typ());
            f(&mut self, new_input_interface, output_interface)
        };

        let update_input = {
            let primitives = old_input_interface.into_primitives();
            primitives
                .into_iter()
                .filter_map(|(interface, path)| interface.get_channel().map(|channel| (path, channel)))
                .collect::<HashMap<_, _>>()
        };
        let update_output = {
            let primitives = old_output_interface.into_primitives();
            primitives
                .into_iter()
                .filter_map(|(interface, path)| {
                    interface.get_channel().map(|channel| {
                        (path, match channel.endpoint() {
                            Endpoint::Input { path } => update_input.get(&path).unwrap().clone(),
                            _ => channel,
                        })
                    })
                })
                .collect::<HashMap<_, _>>()
        };

        // Updates old submodules' interfaces.
        for (_, ref mut interface) in self.submodules.iter_mut().take(old_submodules_len) {
            *interface = interface
                .clone()
                .into_primitives()
                .into_iter()
                .map(|(interface, path)| {
                    (
                        match interface {
                            Interface::Unit => Interface::Unit,
                            Interface::Channel(channel) => Interface::Channel(match channel.endpoint() {
                                Endpoint::Input { path } => update_input.get(&path).unwrap().clone(),
                                _ => channel,
                            }),
                            _ => panic!("internal compiler error"),
                        },
                        path,
                    )
                })
                .collect();
        }
        for (_, ref mut interface) in self.submodules.iter_mut() {
            *interface = interface
                .clone()
                .into_primitives()
                .into_iter()
                .map(|(interface, path)| {
                    (
                        match interface {
                            Interface::Unit => Interface::Unit,
                            Interface::Channel(channel) => Interface::Channel(match channel.endpoint() {
                                Endpoint::Temp { path } => {
                                    let channel = update_output.get(&path).unwrap().clone();
                                    if matches!(channel.endpoint(), Endpoint::Temp { .. }) {
                                        // TODO: Analyze cyclic assignment and handle it
                                        todo!()
                                    }
                                    channel
                                }
                                _ => channel,
                            }),
                            _ => panic!("internal compiler error"),
                        },
                        path,
                    )
                })
                .collect();
        }

        // Updates new output channels.
        let new_output_interface = new_output_interface
            .into_primitives()
            .into_iter()
            .map(|(interface, path)| {
                (
                    match interface {
                        Interface::Unit => Interface::Unit,
                        Interface::Channel(channel) => Interface::Channel(match channel.endpoint() {
                            Endpoint::Temp { path } => {
                                let channel = update_output.get(&path).unwrap().clone();
                                if matches!(channel.endpoint(), Endpoint::Temp { .. }) {
                                    // TODO: Analyze cyclic assignment and handle it
                                    todo!()
                                }
                                channel
                            }
                            _ => channel,
                        }),
                        _ => panic!("internal compiler error"),
                    },
                    path,
                )
            })
            .collect();

        // Wires new input/output channels.
        self.input_interface = input_interface(&iw_interface_typ);
        self.output_interface = new_output_interface;

        self
    }

    /// `and_then` in LIR.
    ///
    /// For more details, please consult `and_then` method in HIR.
    pub fn and_then(self, f: impl FnOnce(Interface, &mut CompositeModule) -> Interface) -> CompositeModule {
        let input_interface_typ = self.input_interface_typ();
        self.wrap(input_interface_typ, |k, i, o| (i, f(o, k)))
    }
}
