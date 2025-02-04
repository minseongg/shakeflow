//! Low-level IR's prelude.

use std::collections::VecDeque;
use std::fmt::Display;
use std::iter::FromIterator;
use std::ops::*;

use crate::utils::join_options;

/// Shape of an array.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    inner: VecDeque<usize>,
}

impl Shape {
    /// Creates new shape.
    pub fn new<I: IntoIterator<Item = usize>>(iterable: I) -> Self {
        Self { inner: iterable.into_iter().collect() }
    }

    /// Returns dimension of array.
    pub fn dim(&self) -> usize {
        self.inner.len()
    }

    /// Returns number of elements in array.
    pub fn width(&self) -> usize {
        self.inner.iter().product()
    }

    /// TODO: Documentation
    pub fn get(&self, index: usize) -> usize {
        assert!(self.dim() > index);
        *self.inner.get(index).unwrap()
    }

    /// TODO: Documentation
    #[must_use]
    pub fn multiple(&self, n: usize) -> Self {
        let mut inner = self.inner.clone();
        let front = inner.pop_front().unwrap();
        inner.push_front(front * n);

        Self { inner }
    }

    /// TODO: Documentation
    #[must_use]
    pub fn divide(&self, n: usize) -> Self {
        let mut inner = self.inner.clone();
        let front = inner.pop_front().unwrap();
        assert_eq!(front % n, 0);
        inner.push_front(front / n);

        Self { inner }
    }
}

/// LIR value type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PortDecls {
    /// Collection of channels.
    Struct(Vec<(Option<String>, PortDecls)>),

    /// Single channel which contains its width.
    Bits(Shape),
}

impl PortDecls {
    /// Width of `PortDecls`.
    pub fn width(&self) -> usize {
        match self {
            PortDecls::Struct(inner) => inner.iter().map(|(_, m)| m.width()).sum(),
            PortDecls::Bits(shape) => shape.width(),
        }
    }

    /// Maximum dimension of the primitive value types in `PortDecls`.
    pub fn max_dim(&self) -> usize {
        self.iter().map(|(_, shape)| shape.dim()).max().unwrap_or(1)
    }

    /// Iterator for `PortDecls`.
    ///
    /// # Note
    ///
    /// The iterator returns (name, width) for inner fields **ONLY** with nonzero width.
    /// This is to ignore meaningless unit types. (e.g. The unit type in `Keep<V, ()>`)
    pub fn iter(&self) -> ValueTypIterator {
        self.into_iter()
    }

    /// Consumes the `PortDecls`, returning new `PortDecls` with width of each field multiplied by `n`.
    #[must_use]
    pub fn multiple(&self, n: usize) -> Self {
        match self {
            PortDecls::Struct(inner) => {
                PortDecls::Struct(inner.clone().into_iter().map(|(name, m)| (name, m.multiple(n))).collect::<Vec<_>>())
            }
            PortDecls::Bits(shape) => PortDecls::Bits(shape.multiple(n)),
        }
    }

    /// Consumes the `PortDecls`, returning new `PortDecls` with width of each field divided by `n`.
    #[must_use]
    pub fn divide(&self, n: usize) -> Self {
        match self {
            PortDecls::Struct(inner) => {
                PortDecls::Struct(inner.clone().into_iter().map(|(name, m)| (name, m.divide(n))).collect::<Vec<_>>())
            }
            PortDecls::Bits(shape) => PortDecls::Bits(shape.divide(n)),
        }
    }

    fn iter_with_prefix(&self, prefix: Option<String>) -> ValueTypIterator {
        let mut iter_vec = vec![];

        match self {
            PortDecls::Struct(inner) => {
                for (name, member) in inner {
                    iter_vec.extend(member.iter_with_prefix(join_options("_", [prefix.clone(), name.clone()])).inner)
                }
            }
            PortDecls::Bits(shape) => {
                if shape.width() > 0 {
                    iter_vec.push((prefix, shape.clone()));
                }
            }
        }

        ValueTypIterator { inner: iter_vec.into() }
    }
}

impl IntoIterator for &PortDecls {
    type IntoIter = ValueTypIterator;
    type Item = (Option<String>, Shape);

    fn into_iter(self) -> Self::IntoIter {
        self.iter_with_prefix(None)
    }
}

/// Iterator for `PortDecls`.
#[derive(Debug)]
pub struct ValueTypIterator {
    inner: VecDeque<(Option<String>, Shape)>,
}

impl Iterator for ValueTypIterator {
    type Item = (Option<String>, Shape);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.pop_front()
    }
}

/// Channel's type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelTyp {
    /// Forward value.
    pub fwd: PortDecls,

    /// Backward value.
    pub bwd: PortDecls,
}

impl ChannelTyp {
    /// Creates a new channel type.
    pub const fn new(fwd: PortDecls, bwd: PortDecls) -> Self {
        Self { fwd, bwd }
    }
}

/// Input/output channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    /// Channel's typ.
    pub typ: ChannelTyp,

    /// Channel's endpoint.
    pub endpoint: Endpoint,
}

impl Channel {
    /// Returns channel type.
    pub fn typ(&self) -> ChannelTyp {
        self.typ.clone()
    }

    /// Returns endpoint.
    pub fn endpoint(&self) -> Endpoint {
        self.endpoint.clone()
    }
}

/// Endpoint's node.
// TODO: Add array range types
#[allow(variant_size_differences)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EndpointNode {
    /// Element of array.
    Index(usize),

    /// Element of expansive array.
    ExpansiveIndex(usize),

    /// Field of struct. The first `String` indicates name of the field, and the second `Option<String>`
    /// indicates separator. If it is `None`, then separator is '_'.
    Field(String, Option<String>),
}

/// Endpoint's path.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct EndpointPath {
    /// List of endpoint nodes.
    pub inner: VecDeque<EndpointNode>,
}

impl FromIterator<EndpointNode> for EndpointPath {
    fn from_iter<T: IntoIterator<Item = EndpointNode>>(iter: T) -> Self {
        Self { inner: iter.into_iter().collect() }
    }
}

impl Deref for EndpointPath {
    type Target = VecDeque<EndpointNode>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for EndpointPath {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Wire's endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Endpoint {
    /// Input interface.
    Input {
        /// Interface's endpoint path in the input.
        path: EndpointPath,
    },

    /// Submodule endpoint.
    Submodule {
        /// Submodule's index in the module's submodules.
        submodule_index: usize,

        /// Interface's endpoint path in the submodule.
        path: EndpointPath,
    },

    /// Temporary interface used in `CompositeModule::wrap`.
    ///
    /// # Note
    ///
    /// This endpoint type does not appear in the final module. This type is only used to replace
    /// the output interface of the inner module and then updated to the original output interface
    /// of the inner module in `CompositeModule::wrap`.
    Temp {
        /// Interface's endpoint path in the output.
        path: EndpointPath,
    },
}

impl Endpoint {
    /// Creates a new endpoint on input.
    pub fn input(path: EndpointPath) -> Self {
        Self::Input { path }
    }

    /// Creates a new endpoint on submodule.
    pub fn submodule(submodule_index: usize, path: EndpointPath) -> Self {
        Self::Submodule { submodule_index, path }
    }

    /// Creates a new temporary endpoint.
    pub fn temp(path: EndpointPath) -> Self {
        Self::Temp { path }
    }

    /// Returns endpoint path.
    pub fn path(&self) -> &EndpointPath {
        match self {
            Endpoint::Input { path } => path,
            Endpoint::Submodule { path, .. } => path,
            Endpoint::Temp { path } => path,
        }
    }
}

/// Unary operators.
// TODO: Add more cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Negation
    Negation,
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            UnaryOp::Negation => "~",
        };

        write!(f, "{}", value)
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Addition
    Add,

    /// Subtraction
    Sub,

    /// Multiplication
    Mul,

    /// Division
    Div,

    /// Modulus
    Mod,

    /// Or (bitwise)
    Or,

    /// And (bitwise)
    And,

    /// Xor (bitwise)
    Xor,

    /// Eq (bitwise, `a ~^ b`)
    Eq,

    /// Eq (arithmetic, `a == b`)
    EqArithmetic,

    /// Less than
    Less,

    /// Greater than
    Greater,

    /// Less than or equal
    LessEq,

    /// Greater than or equal
    GreaterEq,

    /// Shift left
    ShiftLeft,

    /// Shift right
    ShiftRight,
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Or => "|",
            BinaryOp::And => "&",
            BinaryOp::Xor => "^",
            BinaryOp::Eq => "~^",
            BinaryOp::EqArithmetic => "==",
            BinaryOp::Less => "<",
            BinaryOp::Greater => ">",
            BinaryOp::LessEq => "<=",
            BinaryOp::GreaterEq => ">=",
            BinaryOp::ShiftLeft => "<<",
            BinaryOp::ShiftRight => ">>",
        };

        write!(f, "{}", value)
    }
}
