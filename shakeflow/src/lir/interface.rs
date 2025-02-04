//! Interface.

use std::collections::HashMap;

use linked_hash_map::LinkedHashMap;

use super::*;

/// Interface's type.
#[allow(variant_size_differences)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterfaceTyp {
    /// Unit type
    Unit,

    /// Single channel type
    Channel(ChannelTyp),

    /// Array of interface types
    Array(Box<InterfaceTyp>, usize),

    /// Expansive array of interface types
    ExpansiveArray(Box<InterfaceTyp>, usize),

    /// Struct of interface types. The first `String` of value indicates separator of the field.
    Struct(LinkedHashMap<String, (Option<String>, InterfaceTyp)>),
}

impl InterfaceTyp {
    /// Returns forward signal port declarations.
    pub fn fwd(&self) -> PortDecls {
        match self {
            InterfaceTyp::Unit => PortDecls::Bits(Shape::new([0])),
            InterfaceTyp::Channel(typ) => typ.fwd.clone(),
            InterfaceTyp::Array(typ, n) => typ.fwd().multiple(*n),
            InterfaceTyp::ExpansiveArray(typ, n) => {
                PortDecls::Struct((0..*n).map(|i| (Some(i.to_string()), typ.fwd())).collect())
            }
            InterfaceTyp::Struct(inner) => PortDecls::Struct(
                inner
                    .into_iter()
                    .map(|(name, (_sep, typ))| (if name.is_empty() { None } else { Some(name.clone()) }, typ.fwd()))
                    .collect(),
            ),
        }
    }

    /// Returns backward signal port declarations.
    pub fn bwd(&self) -> PortDecls {
        match self {
            InterfaceTyp::Unit => PortDecls::Bits(Shape::new([0])),
            InterfaceTyp::Channel(typ) => typ.bwd.clone(),
            InterfaceTyp::Array(typ, n) => typ.bwd().multiple(*n),
            InterfaceTyp::ExpansiveArray(typ, n) => {
                PortDecls::Struct((0..*n).map(|i| (Some(i.to_string()), typ.bwd())).collect())
            }
            InterfaceTyp::Struct(inner) => PortDecls::Struct(
                inner
                    .into_iter()
                    .map(|(name, (_sep, typ))| (if name.is_empty() { None } else { Some(name.clone()) }, typ.bwd()))
                    .collect(),
            ),
        }
    }

    /// TODO: Documentation
    pub fn get_channel_typ(self) -> Option<ChannelTyp> {
        if let InterfaceTyp::Channel(channel_typ) = self {
            Some(channel_typ)
        } else {
            None
        }
    }

    /// Returns primitive interface types and their endpoint paths in the interface type.
    // TODO: Change return type and consider primitives of `VarArray`.
    pub fn into_primitives(&self) -> Vec<(InterfaceTyp, EndpointPath)> {
        match self {
            InterfaceTyp::Unit | InterfaceTyp::Channel(_) => vec![(self.clone(), EndpointPath::default())],
            InterfaceTyp::Array(interface_typ, count) => (0..*count)
                .flat_map(|i| {
                    interface_typ.into_primitives().into_iter().map(move |(primitive_typ, mut path)| {
                        path.inner.push_front(EndpointNode::Index(i));
                        (primitive_typ, path)
                    })
                })
                .collect(),
            InterfaceTyp::ExpansiveArray(interface_typ, count) => (0..*count)
                .flat_map(|i| {
                    interface_typ.into_primitives().into_iter().map(move |(primitive_typ, mut path)| {
                        path.inner.push_front(EndpointNode::ExpansiveIndex(i));
                        (primitive_typ, path)
                    })
                })
                .collect(),
            InterfaceTyp::Struct(inner) => inner
                .into_iter()
                .flat_map(|(name, (sep, interface_typ))| {
                    interface_typ.into_primitives().into_iter().map(|(primitive_typ, mut path)| {
                        path.inner.push_front(EndpointNode::Field(name.clone(), sep.clone()));
                        (primitive_typ, path)
                    })
                })
                .collect(),
        }
    }

    /// Returns subinterface given a endpoint path
    pub fn get_subinterface(&self, mut path: EndpointPath) -> Self {
        if let Some(front) = path.pop_front() {
            match (front, self) {
                (EndpointNode::Index(i), InterfaceTyp::Array(typ, size)) => {
                    assert!(i < *size);
                    typ.get_subinterface(path)
                }
                (EndpointNode::ExpansiveIndex(i), InterfaceTyp::ExpansiveArray(typ, size)) => {
                    assert!(i < *size);
                    typ.get_subinterface(path)
                }
                (EndpointNode::Field(field, _), InterfaceTyp::Struct(map)) => {
                    if let Some((_, typ)) = map.get(&field) {
                        typ.get_subinterface(path)
                    } else {
                        panic!("{} does not exist in the struct", field)
                    }
                }
                _ => panic!("path and interface doesn't match"),
            }
        } else {
            self.clone()
        }
    }
}

/// Input/output interface.
#[allow(variant_size_differences)]
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum Interface {
    /// Unit
    #[default]
    Unit,

    /// Single channel
    Channel(Channel),

    /// Array of interfaces
    Array(Vec<Interface>),

    /// Expansive array of interfaces
    ExpansiveArray(Vec<Interface>),

    /// Struct of interfaces. The first `Option<String>` of value indicates separator of the field.
    /// If it is `None`, then separator is '_'.
    Struct(LinkedHashMap<String, (Option<String>, Interface)>),
}

impl Interface {
    /// `comb_inline` in LIR.
    ///
    /// For more details, please consult `comb_inline` method in HIR.
    pub fn comb_inline(self, k: &mut CompositeModule, module: Module) -> Interface {
        k.add_submodule(module, self)
    }

    /// `fsm` in LIR.
    ///
    /// For more details, please consult `fsm` method in HIR.
    pub fn fsm(
        self, k: &mut CompositeModule, module_name: Option<&str>, output_interface_typ: InterfaceTyp, s_typ: PortDecls,
        f: impl Fn(ExprId, ExprId, ExprId) -> (ExprId, ExprId, ExprId), init: ExprId,
    ) -> Interface {
        let input_interface_typ = self.typ();

        self.comb_inline(
            k,
            Fsm::new(
                module_name.unwrap_or("fsm").to_string(),
                input_interface_typ,
                output_interface_typ,
                s_typ,
                f,
                init,
            )
            .into(),
        )
    }
}

impl Interface {
    /// TODO: Documentation
    pub fn get_channel(self) -> Option<Channel> {
        if let Interface::Channel(channel) = self {
            Some(channel)
        } else {
            None
        }
    }

    /// Returns the interface type.
    pub fn typ(&self) -> InterfaceTyp {
        match self {
            Interface::Unit => InterfaceTyp::Unit,
            Interface::Channel(channel) => InterfaceTyp::Channel(channel.typ.clone()),
            Interface::Array(inner) => InterfaceTyp::Array(Box::new(inner[0].typ()), inner.len()),
            Interface::ExpansiveArray(inner) => InterfaceTyp::ExpansiveArray(Box::new(inner[0].typ()), inner.len()),
            Interface::Struct(inner) => InterfaceTyp::Struct(
                inner.iter().map(|(name, (sep, interface))| (name.clone(), (sep.clone(), interface.typ()))).collect(),
            ),
        }
    }

    /// Returns primitive interfaces in the interface.
    pub fn into_primitives(&self) -> Vec<(Interface, EndpointPath)> {
        match self {
            Interface::Unit | Interface::Channel(_) => vec![(self.clone(), EndpointPath::default())],
            Interface::Array(interfaces) => interfaces
                .iter()
                .enumerate()
                .flat_map(|(i, interface)| {
                    interface.into_primitives().into_iter().map(move |(primitive, mut path)| {
                        path.inner.push_front(EndpointNode::Index(i));
                        (primitive, path)
                    })
                })
                .collect(),
            Interface::ExpansiveArray(interfaces) => interfaces
                .iter()
                .enumerate()
                .flat_map(|(i, interface)| {
                    interface.into_primitives().into_iter().map(move |(primitive, mut path)| {
                        path.inner.push_front(EndpointNode::ExpansiveIndex(i));
                        (primitive, path)
                    })
                })
                .collect(),
            Interface::Struct(inner) => inner
                .iter()
                .flat_map(|(name, (sep, interface))| {
                    interface.into_primitives().into_iter().map(|(primitive, mut path)| {
                        path.inner.push_front(EndpointNode::Field(name.clone(), sep.clone()));
                        (primitive, path)
                    })
                })
                .collect(),
        }
    }
}

impl FromIterator<(Interface, EndpointPath)> for Interface {
    /// Constructs interface from primitive interfaces.
    fn from_iter<I: IntoIterator<Item = (Interface, EndpointPath)>>(iter: I) -> Self {
        let mut primitives = iter.into_iter().collect::<Vec<_>>();
        assert!(!primitives.is_empty());

        let is_primitive = primitives[0].1.inner.front().is_none();
        if is_primitive {
            assert_eq!(primitives.len(), 1);
            let (primitive, _) = primitives.pop().unwrap();
            assert!(matches!(primitive, Interface::Unit | Interface::Channel(_)));
            primitive
        } else {
            match primitives[0].1.inner.front().unwrap() {
                EndpointNode::Index(_) => {
                    let mut interfaces = HashMap::<usize, Vec<(Interface, EndpointPath)>>::new();
                    for (interface, mut path) in primitives {
                        let node = path.inner.pop_front().unwrap();
                        match node {
                            EndpointNode::Index(i) => {
                                interfaces.entry(i).or_default();
                                let primitives = interfaces.get_mut(&i).unwrap();
                                primitives.push((interface, path));
                            }
                            _ => panic!("internal compiler error"),
                        }
                    }
                    let len = interfaces.len();
                    Interface::Array(
                        (0..len).map(|i| interfaces.get(&i).unwrap().clone().into_iter().collect()).collect(),
                    )
                }
                EndpointNode::ExpansiveIndex(_) => {
                    let mut interfaces = HashMap::<usize, Vec<(Interface, EndpointPath)>>::new();
                    for (interface, mut path) in primitives {
                        let node = path.inner.pop_front().unwrap();
                        match node {
                            EndpointNode::ExpansiveIndex(i) => {
                                interfaces.entry(i).or_default();
                                let primitives = interfaces.get_mut(&i).unwrap();
                                primitives.push((interface, path));
                            }
                            _ => panic!("internal compiler error"),
                        }
                    }
                    let len = interfaces.len();
                    Interface::ExpansiveArray(
                        (0..len).map(|i| interfaces.get(&i).unwrap().clone().into_iter().collect()).collect(),
                    )
                }
                EndpointNode::Field(..) => {
                    let mut inner = LinkedHashMap::<String, (Option<String>, Vec<(Interface, EndpointPath)>)>::new();
                    for (interface, mut path) in primitives {
                        let node = path.inner.pop_front().unwrap();
                        match node {
                            EndpointNode::Field(name, sep) => {
                                inner.entry(name.clone()).or_insert((sep, Vec::new()));
                                let primitives = inner.get_mut(&name).unwrap();
                                primitives.1.push((interface, path));
                            }
                            _ => panic!("internal compiler error"),
                        }
                    }
                    Interface::Struct(
                        inner
                            .into_iter()
                            .map(|(name, (sep, primitives))| (name, (sep, primitives.into_iter().collect())))
                            .collect(),
                    )
                }
            }
        }
    }
}
