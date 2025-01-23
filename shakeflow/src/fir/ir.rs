//! FIRRTL.
//!
//! For simplicity, we omitted some items such as `Info`, `StringLit`, `ExtModule`, ..

use std::fmt::Display;

use crate::codegen::*;
use crate::lir;
use crate::utils::indent;

const INDENT: usize = 2;

/// Primitive operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimOp {
    /// Addition
    Add,

    /// Subtraction
    Sub,

    /// Multiplication
    Mul,

    /// Division
    Div,

    /// Remainder
    Rem,

    /// Less Than
    Lt,

    /// Less Than Or Equal To
    Leq,

    /// Greater Than
    Gt,

    /// Greater Than Or Equal To
    Geq,

    /// Equal To
    Eq,

    /// Not Equal To
    Neq,

    /// Padding
    Pad,

    /// Static Shift Left
    Shl,

    /// Static Shift Right
    Shr,

    /// Dynamic Shift Left
    Dshl,

    /// Dynamic Shift Right
    Dshr,

    /// Bitwise Complement
    Not,

    /// Bitwise And
    And,

    /// Bitwise Or
    Or,

    /// Bitwise Exclusive Or
    Xor,

    /// Bitwise And Reduce
    Andr,

    /// Bitwise Or Reduce
    Orr,

    /// Bitwise Exclusive Or Reduce
    Xorr,

    /// Concatenate
    Cat,

    /// Bit Extraction
    Bits,

    /// Head
    Head,

    /// Tail
    Tail,
}

impl Display for PrimOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            PrimOp::Add => "add",
            PrimOp::Sub => "sub",
            PrimOp::Mul => "mul",
            PrimOp::Div => "div",
            PrimOp::Rem => "rem",
            PrimOp::Lt => "lt",
            PrimOp::Leq => "leq",
            PrimOp::Gt => "gt",
            PrimOp::Geq => "geq",
            PrimOp::Eq => "eq",
            PrimOp::Neq => "neq",
            PrimOp::Pad => "pad",
            PrimOp::Shl => "shl",
            PrimOp::Shr => "shr",
            PrimOp::Dshl => "dshl",
            PrimOp::Dshr => "dshr",
            PrimOp::Not => "not",
            PrimOp::And => "and",
            PrimOp::Or => "or",
            PrimOp::Xor => "xor",
            PrimOp::Andr => "andr",
            PrimOp::Orr => "orr",
            PrimOp::Xorr => "xorr",
            PrimOp::Cat => "cat",
            PrimOp::Bits => "bits",
            PrimOp::Head => "head",
            PrimOp::Tail => "tail",
        };

        write!(f, "{}", value)
    }
}

impl From<lir::UnaryOp> for PrimOp {
    fn from(op: lir::UnaryOp) -> Self {
        match op {
            lir::UnaryOp::Negation => PrimOp::Not,
        }
    }
}

impl From<lir::BinaryOp> for PrimOp {
    fn from(op: lir::BinaryOp) -> Self {
        match op {
            lir::BinaryOp::Add => PrimOp::Add,
            lir::BinaryOp::Sub => PrimOp::Sub,
            lir::BinaryOp::Mul => PrimOp::Mul,
            lir::BinaryOp::Div => PrimOp::Div,
            lir::BinaryOp::Mod => PrimOp::Rem,
            lir::BinaryOp::Or => PrimOp::Or,
            lir::BinaryOp::And => PrimOp::And,
            lir::BinaryOp::Xor => PrimOp::Xor,
            lir::BinaryOp::Eq => todo!(),
            lir::BinaryOp::EqArithmetic => PrimOp::Eq,
            lir::BinaryOp::Less => PrimOp::Lt,
            lir::BinaryOp::Greater => PrimOp::Gt,
            lir::BinaryOp::LessEq => PrimOp::Leq,
            lir::BinaryOp::GreaterEq => PrimOp::Geq,
            lir::BinaryOp::ShiftLeft => PrimOp::Shl,
            lir::BinaryOp::ShiftRight => PrimOp::Shr,
        }
    }
}

impl PrimOp {
    /// Returns true if `self` is comparison operator.
    #[inline]
    pub fn is_comparison(self) -> bool {
        matches!(self, PrimOp::Lt | PrimOp::Leq | PrimOp::Gt | PrimOp::Geq | PrimOp::Eq | PrimOp::Neq)
    }

    /// Returns true if `self` is binary bitwise operator.
    #[inline]
    pub fn is_binary_bitwise(self) -> bool { matches!(self, PrimOp::And | PrimOp::Or | PrimOp::Xor) }

    /// Returns true if `self` is bitwise reduction operator.
    #[inline]
    pub fn is_bitwise_reduction(self) -> bool { matches!(self, PrimOp::Andr | PrimOp::Orr | PrimOp::Xorr) }
}

/// Expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    /// Previously declared circuit component.
    Reference {
        /// Name of the component
        name: String,
    },
    /// Sub-element of an expression with a bundle type.
    SubField {
        /// Input signal
        expr: Box<Expression>,
        /// Name of the field
        name: String,
    },
    /// Sub-element of an expression with a vector type.
    SubIndex {
        /// Input signal
        expr: Box<Expression>,
        /// Index of the element
        value: usize,
    },
    /// Sub-element of a vector-typed expression using a calculated index.
    SubAccess {
        /// Input signal
        expr: Box<Expression>,
        /// Index of the element
        index: Box<Expression>,
    },
    /// One of two input expressions depending on the value of an unsigned selection signal.
    Mux {
        /// Selection signal
        cond: Box<Expression>,
        /// Selected signal when `cond` is true
        tval: Box<Expression>,
        /// Selected signal when `cond` is false
        fval: Box<Expression>,
    },
    /// Input expression guarded with an unsigned single bit valid signal.
    ValidIf {
        /// Valid signal
        cond: Box<Expression>,
        /// Input signal
        valid: Box<Expression>,
    },
    /// Literal.
    Literal {
        /// Value represented in binary format
        value: String,
        /// Width
        width: Option<usize>,
    },
    /// Primitive operation.
    DoPrim {
        /// Primitive operator
        op: PrimOp,
        /// Arguments
        args: Vec<Expression>,
        /// Constants
        consts: Vec<usize>,
    },
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Reference { name } => write!(f, "{}", name.clone()),
            Expression::SubField { expr, name } => {
                write!(f, "{}.{}", expr, name)
            }
            Expression::SubIndex { expr, value } => {
                write!(f, "{}[{}]", expr, value)
            }
            Expression::SubAccess { expr, index } => {
                write!(f, "{}[{}]", expr, index)
            }
            Expression::Mux { cond, tval, fval } => {
                write!(f, "mux({}, {}, {})", cond, tval, fval)
            }
            Expression::ValidIf { cond, valid } => {
                write!(f, "validif({}, {})", cond, valid)
            }
            Expression::Literal { value, width } => write!(
                f,
                "UInt{}({})",
                match width {
                    None => "".to_string(),
                    Some(width) => format!("<{}>", width),
                },
                value
            ),
            Expression::DoPrim { op, args, consts } => {
                write!(
                    f,
                    "{}({})",
                    op,
                    ::std::iter::empty()
                        .chain(args.iter().map(|s| s.to_string()))
                        .chain(consts.iter().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
    }
}

impl From<usize> for Expression {
    fn from(n: usize) -> Self { Expression::Literal { value: n.to_string(), width: None } }
}

#[allow(clippy::should_implement_trait)]
impl Expression {
    /// Reference expression.
    #[inline]
    pub fn reference(name: String) -> Self { Expression::Reference { name } }

    /// Returns true if `self` is reference.
    #[inline]
    pub fn is_reference(&self) -> bool { matches!(self, Expression::Reference { .. }) }

    /// Clock expression.
    #[inline]
    pub fn clk() -> Self { Expression::reference("clk".to_string()) }

    /// Reset expression.
    #[inline]
    pub fn rst() -> Self { Expression::reference("rst".to_string()) }

    /// Subfield expression.
    #[inline]
    pub fn sub_field(expr: Expression, name: String) -> Self { Expression::SubField { expr: Box::new(expr), name } }

    /// Subindex expression.
    #[inline]
    pub fn sub_index(expr: Expression, value: usize) -> Self { Expression::SubIndex { expr: Box::new(expr), value } }

    /// Subaccess expression.
    #[inline]
    pub fn sub_access(expr: Expression, index: Expression) -> Self {
        Expression::SubAccess { expr: Box::new(expr), index: Box::new(index) }
    }

    /// Mux expression.
    #[inline]
    pub fn mux(cond: Expression, tval: Expression, fval: Expression) -> Self {
        Expression::Mux { cond: Box::new(cond), tval: Box::new(tval), fval: Box::new(fval) }
    }

    /// Valid-if expression.
    #[inline]
    pub fn valid_if(cond: Expression, valid: Expression) -> Self {
        Expression::ValidIf { cond: Box::new(cond), valid: Box::new(valid) }
    }

    /// Literal expression.
    ///
    /// Value should be represented in binary format. (e.g. "01001")
    #[inline]
    pub fn literal(value: LogicValues, width: Option<usize>) -> Self {
        Expression::Literal { value: format!("\"b{}\"", value), width }
    }

    /// Primitive operation.
    #[inline]
    pub fn do_prim(op: PrimOp, args: Vec<Expression>, consts: Vec<usize>) -> Self {
        Expression::DoPrim { op, args, consts }
    }

    /// Add operation.
    ///
    /// Result width is `max(w_e1, w_e2) + 1`.
    #[inline]
    pub fn add(e1: Self, e2: Self) -> Self { Expression::do_prim(PrimOp::Add, vec![e1, e2], Vec::new()) }

    /// Subtract operation.
    ///
    /// Result width is `max(w_e1, w_e2) + 1`.
    #[inline]
    pub fn sub(e1: Self, e2: Self) -> Self { Expression::do_prim(PrimOp::Sub, vec![e1, e2], Vec::new()) }

    /// Multiply operation.
    ///
    /// Result width is `w_e1 + w_e2`.
    #[inline]
    pub fn mul(e1: Self, e2: Self) -> Self { Expression::do_prim(PrimOp::Mul, vec![e1, e2], Vec::new()) }

    /// Divide operation.
    ///
    /// Result width is `w_num`.
    #[inline]
    pub fn div(num: Self, den: Self) -> Self { Expression::do_prim(PrimOp::Div, vec![num, den], Vec::new()) }

    /// Modulus operation.
    ///
    /// Result width is `min(w_num, w_den)`.
    #[inline]
    pub fn rem(num: Self, den: Self) -> Self { Expression::do_prim(PrimOp::Rem, vec![num, den], Vec::new()) }

    /// Comparison operations. (lt, leq, gt, geq, eq, neq)
    ///
    /// Result width is `1`.
    #[inline]
    pub fn cmp(op: PrimOp, e1: Self, e2: Self) -> Self {
        assert!(op.is_comparison());
        Expression::do_prim(op, vec![e1, e2], Vec::new())
    }

    /// Padding operation.
    ///
    /// Result width is `max(w_e, n)`.
    #[inline]
    pub fn pad(e: Self, n: usize) -> Self { Expression::do_prim(PrimOp::Pad, vec![e], vec![n]) }

    /// Shift left operation.
    ///
    /// Result width is `w_e + n`.
    #[inline]
    pub fn shl(e: Self, n: usize) -> Self { Expression::do_prim(PrimOp::Shl, vec![e], vec![n]) }

    /// Shift right operation.
    ///
    /// Result width is `max(w_e - n, 1)`.
    #[inline]
    pub fn shr(e: Self, n: usize) -> Self { Expression::do_prim(PrimOp::Shr, vec![e], vec![n]) }

    /// Dynamic shift left operation.
    ///
    /// Result width is `w_e1 + 2 ^ w_e2 - 1`.
    #[inline]
    pub fn dshl(e1: Self, e2: Self) -> Self { Expression::do_prim(PrimOp::Dshl, vec![e1, e2], Vec::new()) }

    /// Dynamic shift right operation.
    ///
    /// Result width is `w_e1`.
    #[inline]
    pub fn dshr(e1: Self, e2: Self) -> Self { Expression::do_prim(PrimOp::Dshr, vec![e1, e2], Vec::new()) }

    /// Bitwise complement operation.
    ///
    /// Result width is `w_e`.
    #[inline]
    pub fn not(e: Self) -> Self { Expression::do_prim(PrimOp::Not, vec![e], Vec::new()) }

    /// Binary bitwise operations. (and, or, xor)
    ///
    /// Result width is `max(w_e1, w_e2)`.
    #[inline]
    pub fn binary_bitwise(op: PrimOp, e1: Self, e2: Self) -> Self {
        assert!(op.is_binary_bitwise());
        Expression::do_prim(op, vec![e1, e2], Vec::new())
    }

    /// Bitwise reduction operations. (andr, orr, xorr)
    ///
    /// Result width is `1`.
    #[inline]
    pub fn bitwise_reduction(op: PrimOp, e: Self) -> Self {
        assert!(op.is_bitwise_reduction());
        Expression::do_prim(op, vec![e], Vec::new())
    }

    /// Concatenate operation.
    ///
    /// Result width is `w_e1 + w_e2`.
    #[inline]
    pub fn cat(e1: Self, e2: Self) -> Self { Expression::do_prim(PrimOp::Cat, vec![e1, e2], Vec::new()) }

    /// Bit extraction operation.
    ///
    /// Result width is `hi - lo + 1`.
    #[inline]
    pub fn bits(e: Self, hi: usize, lo: usize) -> Self { Expression::do_prim(PrimOp::Bits, vec![e], vec![hi, lo]) }

    /// Head.
    ///
    /// Result width is `n`.
    #[inline]
    pub fn head(e: Self, n: usize) -> Self { Expression::do_prim(PrimOp::Head, vec![e], vec![n]) }

    /// Tail.
    ///
    /// Result width is `w_e - n`.
    #[inline]
    pub fn tail(e: Self, n: usize) -> Self { Expression::do_prim(PrimOp::Tail, vec![e], vec![n]) }
}

/// Statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    /// Wire definition.
    DefWire {
        /// Name of the wire
        name: String,
        /// Type of the signal
        tpe: Type,
    },
    /// Register definition.
    DefRegister {
        /// Name of the register
        name: String,
        /// Type of the signal
        tpe: Type,
        /// Clock signal
        clock: Expression,
        /// Reset signal
        reset: Expression,
        /// Initialization signal
        init: Expression,
    },
    /// Module instantiation.
    DefInstance {
        /// Name of the instance
        name: String,
        /// Name of the module
        module: String,
    },
    /// Intermediate value.
    DefNode {
        /// Name of the value
        name: String,
        /// Value
        value: Expression,
    },
    /// Conditional statement.
    Conditionally {
        /// Predicate signal
        pred: Expression,
        /// Then statement
        conseq: Box<Statement>,
        /// Else statement
        alt: Box<Statement>,
    },
    /// Block of statements.
    Block {
        /// Statements
        stmts: Vec<Statement>,
    },
    /// Physically wired connection between two circuit components.
    PartialConnect {
        /// L-value
        loc: Expression,
        /// R-value
        expr: Expression,
    },
    /// Physically wired connection between two circuit components.
    ///
    /// # Note
    ///
    /// Difference between `PartialConnect` is that `PartialConnect` enforces fewer restrictions on
    /// the types and widths of the circuit components it connects.
    ///
    /// For more details, see FIRRTL spec.
    Connect {
        /// L-value
        loc: Expression,
        /// R-value
        expr: Expression,
    },
    /// Indicate that a circuit component contains indeterminate value.
    IsInvalid {
        /// Expression
        expr: Expression,
    },
    /// Empty statement.
    EmptyStmt,
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Statement::DefWire { name, tpe } => {
                write!(f, "wire {} : {}", name, tpe)
            }
            Statement::DefRegister { name, tpe, clock, reset, init } => {
                write!(
                    f,
                    "reg {} : {}, {} with :\n{}",
                    name,
                    tpe,
                    clock,
                    indent(format!("reset => ({}, {})", reset, init), INDENT)
                )
            }
            Statement::DefInstance { name, module } => {
                write!(f, "inst {} of {}", name, module)
            }
            Statement::DefNode { name, value } => {
                write!(f, "node {} = {}", name, value)
            }
            Statement::Conditionally { pred, conseq, alt } => {
                write!(
                    f,
                    "when {} :\n{}{}",
                    pred,
                    indent(conseq.to_string(), INDENT),
                    if matches!(**alt, Statement::EmptyStmt) {
                        "".to_string()
                    } else {
                        format!("\nelse :\n{}", indent(alt.to_string(), INDENT))
                    }
                )
            }
            Statement::Block { stmts } => {
                let res = stmts.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n");

                if res.is_empty() {
                    write!(f, "{}", Statement::EmptyStmt)
                } else {
                    write!(f, "{}", res)
                }
            }
            Statement::PartialConnect { loc, expr } => {
                write!(f, "{} <- {}", loc, expr)
            }
            Statement::Connect { loc, expr } => {
                write!(f, "{} <= {}", loc, expr)
            }
            Statement::IsInvalid { expr } => {
                write!(f, "{} is invalid", expr)
            }
            Statement::EmptyStmt => write!(f, "skip"),
        }
    }
}

impl Statement {
    /// Creates new wire definition.
    #[inline]
    pub fn def_wire(name: String, tpe: Type) -> Self { Statement::DefWire { name, tpe } }

    /// Creates new reg definition.
    #[inline]
    pub fn def_reg(name: String, tpe: Type, init: Expression) -> Self {
        Statement::DefRegister { name, tpe, clock: Expression::clk(), reset: Expression::rst(), init }
    }

    /// Creates new module instantiation.
    #[inline]
    pub fn def_inst(name: String, module: String) -> Self { Statement::DefInstance { name, module } }

    /// Creates new node definition.
    #[inline]
    pub fn def_node(name: String, value: Expression) -> Self { Statement::DefNode { name, value } }

    /// Creates new block statement.
    #[inline]
    pub fn block(stmts: Vec<Statement>) -> Self { Statement::Block { stmts } }

    /// Creates new connect statement.
    #[inline]
    pub fn connect(loc: Expression, expr: Expression) -> Self { Statement::Connect { loc, expr } }
}

/// Type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// Clock type.
    ClockType,
    /// Unsigned integer type.
    UIntType(usize),
    /// Vector type.
    VectorType {
        /// Width of the element
        width: usize,
        /// Number of elements
        size: usize,
    },
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::ClockType => write!(f, "Clock"),
            Type::UIntType(width) => write!(f, "UInt<{}>", width),
            Type::VectorType { width, size } => write!(f, "UInt<{}>[{}]", width, size),
        }
    }
}

impl Type {
    /// Creates new clock type.
    #[inline]
    pub fn clock() -> Self { Type::ClockType }

    /// Creates new unsigned integer type.
    #[inline]
    pub fn uint(width: usize) -> Self { Type::UIntType(width) }

    /// Creates new vector type.
    #[inline]
    pub fn vector(width: usize, size: usize) -> Self { Type::VectorType { width, size } }
}

/// Port of module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Port {
    /// Name of the port
    pub name: String,
    /// Direction of the port
    pub direction: Direction,
    /// Type of the port
    pub tpe: Type,
}

impl Display for Port {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} : {}", self.direction, self.name, self.tpe)
    }
}

impl Port {
    /// Creates new input port.
    pub fn input(name: String, tpe: Type) -> Self { Port { name, direction: Direction::Input, tpe } }

    /// Creates new output port.
    pub fn output(name: String, tpe: Type) -> Self { Port { name, direction: Direction::Output, tpe } }
}

/// An instantiable hardware block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    /// Name of the module
    pub name: String,
    /// Ports of the module
    pub ports: Vec<Port>,
    /// Body of the module
    pub body: Statement,
}

impl Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "module {} :\n{}\n\n{}",
            self.name,
            indent(self.ports.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n"), INDENT),
            indent(self.body.to_string(), INDENT)
        )
    }
}

/// Circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Circuit {
    /// Inner modules
    pub modules: Vec<Module>,
    /// Name of the circuit
    pub main: String,
}

impl Display for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "circuit {} :\n{}\n",
            self.main,
            self.modules.iter().map(|s| s.to_string()).map(|s| indent(s, INDENT)).collect::<Vec<_>>().join("\n")
        )
    }
}
