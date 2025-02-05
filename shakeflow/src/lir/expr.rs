//! LIR Expr.

use std::cell::RefCell;
use std::cmp::Ordering;

use hashcons::merkle::Merkle;

use super::*;
use crate::utils::*;

/// Expr Id
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(usize);

impl ExprId {
    /// Allocates expr to the table and returns the id
    pub fn alloc_expr(expr: Merkle<Expr>) -> Self {
        TABLE.with(|table| table.push(expr))
    }

    /// Returns expr corresponding to given id
    pub fn into_expr(self) -> Merkle<Expr> {
        TABLE.with(|table| table.get(self))
    }
}

/// Expr Table
#[derive(Default)]
pub struct Table {
    inner: RefCell<Vec<Merkle<Expr>>>,

    /// For some trait methods in `std::ops`, they take reference of input value and return reference
    /// of output value which have same lifetime with input value. For example:
    ///
    /// - [`std::ops::Index`](https://doc.rust-lang.org/std/ops/trait.Index.html)
    /// - [`std::ops::Deref`](https://doc.rust-lang.org/std/ops/trait.Deref.html)
    ///
    /// However, there is no safe way to get the reference of output value because it can be the output
    /// value does not exist in the table before the method is called. Therefore, we implemented such
    /// traits by (1) create the output value in the method and (2) store the created value in this
    /// storage and (3) return the reference of the value in the heap storage by unsafe typecasting.
    ///
    /// Since this storage is dropped after the target code is generated, it is safe to use it in
    /// the target code generation.
    pub(crate) storage: RefCell<Vec<Box<dyn TableStorageElement>>>,
}

impl std::fmt::Debug for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table").field("inner", &self.inner).finish()
    }
}

impl Table {
    /// Inserts expr into table.
    fn get(&self, id: ExprId) -> Merkle<Expr> {
        self.inner.borrow().get(id.0).expect("does not have element!").clone()
    }

    /// Returns expr from table by using id.
    fn push(&self, expr: Merkle<Expr>) -> ExprId {
        let id = self.inner.borrow().len();
        self.inner.borrow_mut().push(expr);
        ExprId(id)
    }
}

thread_local! {
    /// Expr Table
    pub(crate) static TABLE: Table = Table::default();
}

#[doc(hidden)]
pub trait TableStorageElement {}

/// Exprs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Don't-care value
    X {
        /// Value type of the expr
        typ: PortDecls,
    },

    /// Constant value
    Constant {
        /// Bitvector constant
        bits: Vec<bool>,

        /// Value type of the expr
        typ: PortDecls,
    },

    /// Repeated expr
    Repeat {
        /// The repeated expr
        inner: ExprId,

        /// Repeat count
        count: usize,
    },

    /// The input expr
    Input {
        /// Name of the expr (e.g. "in", "out", "st" for input, output, state expr of fsm)
        name: Option<String>,

        /// Value type of the expr
        typ: PortDecls,
    },

    /// Member of expr
    Member {
        /// The inner expr
        inner: ExprId,

        /// Index of the member
        index: usize,
    },

    /// Combine exprs
    Struct {
        /// The inner exprs
        inner: Vec<(Option<String>, ExprId)>,
    },

    /// Resize by, e.g., implicit casting
    Resize {
        /// The inner expr
        inner: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,

        /// A new size of the array
        count: usize,
    },

    /// Left shift: `inner << rhs`
    LeftShift {
        /// The inner expr
        inner: ExprId,

        /// Shift amount
        rhs: ExprId,
    },

    /// Right shift: `inner >> rhs`
    RightShift {
        /// The inner expr
        inner: ExprId,

        /// Shift amount
        rhs: ExprId,
    },

    /// Logical negation: `!inner`
    Not {
        /// The input expr
        inner: ExprId,
    },

    /// Binary operation: `op lhs rhs`
    BinaryOp {
        /// Operator
        op: BinaryOp,

        /// Lhs
        lhs: ExprId,

        /// Rhs
        rhs: ExprId,
    },

    /// Fold (bitwise)
    Fold {
        /// The inner expr
        inner: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,

        /// Fold operator
        func: ExprId,

        /// Fold initial value
        init: ExprId,

        /// Fold accumulator
        acc: ExprId,

        /// Inner slice
        inner_slice: ExprId,
    },

    /// Tree Fold
    TreeFold {
        /// The inner expr
        inner: ExprId,

        /// acc
        acc: ExprId,

        /// op
        op: ExprId,

        /// lhs,
        lhs: ExprId,

        /// rhs,
        rhs: ExprId,
    },

    /// Mapped expr
    Map {
        /// The inner expr
        inner: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,

        /// Map function
        func: ExprId,
    },

    /// Indexing: `inner[index]`
    Get {
        /// The inner expr
        inner: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,

        /// Index
        index: ExprId,
    },

    /// Clip: `inner[from..to]`
    Clip {
        /// The inner expr
        inner: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,

        /// Starting index
        from: ExprId,

        /// Array size
        size: usize,
    },

    /// Append
    Append {
        /// Lhs
        lhs: ExprId,

        /// Rhs
        rhs: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,
    },

    /// Zip exprs.
    Zip {
        /// Inner exprs.
        inner: Vec<ExprId>,

        /// Value type of the element exprs.
        typ_inner: Vec<PortDecls>,
    },
    /// Concatenate (2-dimensional to 1-dimensional)
    Concat {
        /// The inner expr
        inner: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,
    },

    /// Chunk (1-dimensional to 2-dimensional)
    Chunk {
        /// The inner expr
        inner: ExprId,

        /// Chunk size
        chunk_size: usize,
    },

    /// Flatten to 1-dimensional
    Repr {
        /// The inner expr
        inner: ExprId,
    },

    /// Sum of exprs
    ///
    /// Note: Although having the `lir::Expr` enum have a `Sum` member is API-wise unintuitive
    ///       (as `.sum()` is often considered to be a hir-level operation instead of lir-level),
    ///       we set it regardless because it needs to be parsed at codegen-level to create an
    ///       actual for-loop instead of a manually unrolled loop.
    Sum {
        /// The inner expr
        inner: ExprId,

        /// Width of a summand
        width_elt: usize,
    },

    /// Conditional operator: `if cond begin lhs end else rhs end`
    Cond {
        /// The condition expr
        cond: ExprId,

        /// Output when the condition is true
        lhs: ExprId,

        /// Output when the condition is false
        rhs: ExprId,
    },

    /// TODO: Documentation
    Set {
        /// The inner expr
        inner: ExprId,

        /// Index of the element
        index: ExprId,

        /// The value after the change
        elt: ExprId,
    },

    /// TODO: Documentation
    SetRange {
        /// The inner expr
        inner: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,

        /// Index of the element
        index: ExprId,

        /// The value after the change
        elts: ExprId,
    },

    /// TODO: Documentation
    GetVarArray {
        /// The inner expr
        inner: ExprId,

        /// Value type of the element expr
        typ_elt: PortDecls,

        /// Index
        index: ExprId,
    },

    /// TODO: Documentation
    SetVarArray {
        /// The inner expr
        inner: ExprId,

        /// Index of the element
        index: ExprId,

        /// The value after the change
        elt: ExprId,
    },

    /// Case statement for verilog. Should always contain a default value.
    Case {
        /// The case expression
        case_expr: ExprId,

        /// Vec of (case, assignment) pairs; doesn't need to be constant
        case_items: Vec<(ExprId, ExprId)>,

        /// Default expr
        default: Option<ExprId>,
    },

    /// Verilog function call.
    Call {
        /// Function name.
        func_name: String,

        /// Arguments.
        args: Vec<ExprId>,

        /// Return value type.
        typ: PortDecls,
    },

    /// `[Expr<V>; N]` to `Expr<[V; N]>`
    ConcatArray {
        /// length N vector of exprs with type elt_typ
        inner: Vec<ExprId>,

        /// elemtent tyoe
        elt_typ: PortDecls,
    },
}

impl Expr {
    /// Type of the expr.
    pub fn port_decls(&self) -> PortDecls {
        match self {
            Self::X { typ } => typ.clone(),
            Self::Constant { typ, .. } => typ.clone(),
            Self::Repeat { inner, count } => inner.into_expr().port_decls().multiple(*count),
            Self::Input { typ, .. } => typ.clone(),
            Self::Member { inner, index } => match inner.into_expr().port_decls() {
                PortDecls::Struct(inner) => inner[*index].clone().1,
                PortDecls::Bits(_) => panic!("Cannot index a `PortDecls::Bits`."),
            },
            Self::Struct { inner } => PortDecls::Struct(
                inner.iter().map(|(name, member)| (name.clone(), member.into_expr().port_decls())).collect(),
            ),
            Self::Resize { typ_elt, count, .. } => typ_elt.multiple(*count),
            Self::LeftShift { .. } | Self::RightShift { .. } | Self::BinaryOp { .. } => PortDecls::uint(self.width()),
            Self::Chunk { inner, .. } => inner.into_expr().port_decls(),
            Self::Not { inner } => inner.into_expr().port_decls(),
            Self::Fold { init, .. } => init.into_expr().port_decls(),
            Self::TreeFold { lhs, .. } => lhs.into_expr().port_decls(),
            Self::Clip { inner, from: _, size, typ_elt } => {
                inner.into_expr().port_decls().divide(inner.into_expr().width() / typ_elt.width()).multiple(*size)
            }
            Self::Append { lhs, rhs, typ_elt } => {
                let count = (lhs.into_expr().width() + rhs.into_expr().width()) / typ_elt.width();
                typ_elt.multiple(count)
            }
            Self::Get { typ_elt, .. } => typ_elt.clone(),
            Self::Repr { inner } => PortDecls::uint(inner.into_expr().width()),
            Self::Map { inner, typ_elt, func } => {
                let count = inner.into_expr().width() / typ_elt.width();
                let func_typ = func.into_expr().port_decls();
                func_typ.multiple(count)
            }
            Self::Zip { inner, .. } => PortDecls::Struct(
                inner
                    .iter()
                    .enumerate()
                    .map(|(idx, expr_id)| (Some(idx.to_string()), expr_id.into_expr().port_decls()))
                    .collect(),
            ),
            Self::Concat { inner, typ_elt } => {
                let count = inner.into_expr().width() / typ_elt.width();
                typ_elt.multiple(count)
            }
            Self::Sum { width_elt, .. } => PortDecls::uint(*width_elt),
            Self::Cond { lhs, rhs, .. } => {
                let lhs_typ = lhs.into_expr().port_decls();
                let rhs_typ = rhs.into_expr().port_decls();
                assert_eq!(lhs_typ, rhs_typ);
                lhs_typ
            }
            Self::Set { inner, .. } => inner.into_expr().port_decls(),
            Self::SetRange { inner, .. } => inner.into_expr().port_decls(),
            Self::GetVarArray { typ_elt, .. } => typ_elt.clone(),
            Self::SetVarArray { inner, .. } => inner.into_expr().port_decls(),
            Self::Case { case_items, default, .. } => {
                if case_items.is_empty() {
                    // If there are no cases, there must be a default case
                    default.as_ref().unwrap().into_expr().port_decls()
                } else {
                    let typ = case_items[0].1.into_expr().port_decls();
                    assert!(case_items.iter().all(|expr| expr.1.into_expr().port_decls() == typ));
                    if let Some(default) = &default {
                        assert_eq!(default.into_expr().port_decls(), typ);
                    }
                    typ
                }
            }
            Self::Call { typ, .. } => typ.clone(),
            Self::ConcatArray { inner, elt_typ } => elt_typ.multiple(inner.len()),
        }
    }

    /// Computes width of the expr.
    // TODO: Memoization?
    pub fn width(&self) -> usize {
        match self {
            Self::X { typ } => typ.width(),
            Self::Constant { bits, .. } => bits.len(),
            Self::BinaryOp { op, lhs, rhs } => match op {
                BinaryOp::And | BinaryOp::Or | BinaryOp::Xor | BinaryOp::Sub => {
                    let lhs_width = lhs.into_expr().width();
                    let rhs_width = rhs.into_expr().width();
                    assert_eq!(lhs_width, rhs_width);
                    lhs_width
                }
                BinaryOp::Add => {
                    let lhs_width = lhs.into_expr().width();
                    let rhs_width = rhs.into_expr().width();
                    assert_eq!(lhs_width, rhs_width);
                    lhs_width + 1
                }
                BinaryOp::Mul => lhs.into_expr().width() + rhs.into_expr().width(),
                BinaryOp::Div => lhs.into_expr().width(),
                BinaryOp::Mod => rhs.into_expr().width(),
                BinaryOp::EqArithmetic
                | BinaryOp::Less
                | BinaryOp::Greater
                | BinaryOp::LessEq
                | BinaryOp::GreaterEq => {
                    let lhs_width = lhs.into_expr().width();
                    let rhs_width = rhs.into_expr().width();
                    assert_eq!(lhs_width, rhs_width);
                    1
                }
                _ => todo!("Unimplemented width for binary operator {:#?}", op),
            },
            Self::Member { inner, index } => {
                let inner_typ = inner.into_expr().port_decls();
                match inner_typ {
                    PortDecls::Struct(inner) => inner[*index].1.width(),
                    PortDecls::Bits(_) => panic!("Cannot index a `PortDecls::Bits`."),
                }
            }
            Self::Concat { inner, .. } => inner.into_expr().width(),
            Self::Map { inner, typ_elt, func } => {
                let inner_width = inner.into_expr().width();
                assert_eq!(inner_width % typ_elt.width(), 0);
                (inner_width / typ_elt.width()) * func.into_expr().width()
            }
            Self::Repeat { inner, count } => inner.into_expr().width() * count,
            Self::Input { typ, .. } => typ.width(),
            Self::Resize { typ_elt, count, .. } => typ_elt.width() * count,
            Self::Not { inner } => inner.into_expr().width(),
            Self::Cond { cond, lhs, rhs } => {
                let cond_width = cond.into_expr().width();
                let lhs_width = lhs.into_expr().width();
                let rhs_width = rhs.into_expr().width();

                assert_eq!(cond_width, 1);
                assert_eq!(lhs_width, rhs_width);
                lhs_width
            }
            Self::LeftShift { inner, .. } => inner.into_expr().width(),
            Self::RightShift { inner, .. } => inner.into_expr().width(),
            Self::Chunk { inner, .. } => inner.into_expr().width(),
            Self::Get { typ_elt, .. } => typ_elt.width(),
            Self::Clip { size, typ_elt, .. } => typ_elt.width() * (size),
            Self::Append { lhs, rhs, .. } => lhs.into_expr().width() + rhs.into_expr().width(),
            Self::Zip { inner, .. } => inner.iter().map(|expr_id| expr_id.into_expr().width()).sum(),
            Self::Repr { inner } => inner.into_expr().width(),
            Self::Struct { inner } => inner.iter().map(|(_, inner)| inner.into_expr().width()).sum(),
            Self::Set { inner, .. } => inner.into_expr().width(),
            Self::SetRange { inner, .. } => inner.into_expr().width(),
            Self::Fold { init, .. } => init.into_expr().width(),
            Self::TreeFold { lhs, .. } => lhs.into_expr().width(),
            Self::Sum { width_elt, .. } => *width_elt,
            Self::GetVarArray { typ_elt, .. } => typ_elt.width(),
            Self::SetVarArray { inner, .. } => inner.into_expr().width(),
            Self::Case { case_items, default, .. } => {
                if case_items.is_empty() {
                    // If there are no cases, there must be a default case
                    default.as_ref().unwrap().into_expr().width()
                } else {
                    let width = case_items[0].1.into_expr().width();
                    assert!(case_items.iter().all(|expr| expr.1.into_expr().width() == width));
                    if let Some(default) = &default {
                        assert_eq!(default.into_expr().width(), width);
                    }
                    width
                }
            }
            Self::Call { typ, .. } => typ.width(),
            Self::ConcatArray { inner, elt_typ } => elt_typ.width() * inner.len(),
        }
    }
}

impl From<Expr> for ExprId {
    fn from(inner: Expr) -> Self {
        ExprId::alloc_expr(Merkle::new(inner))
    }
}

impl Expr {
    /// Don't care value.
    pub fn x(typ: PortDecls) -> ExprId {
        Expr::X { typ }.into()
    }

    /// Input expr.
    pub fn input(typ: PortDecls, name: Option<String>) -> ExprId {
        Expr::Input { name, typ }.into()
    }

    /// Member of input expr.
    pub fn member(input: ExprId, index: usize) -> ExprId {
        Expr::Member { inner: input, index }.into()
    }

    /// Projection.
    pub fn proj(struct_typ: PortDecls, input: ExprId) -> Vec<(Option<String>, ExprId)> {
        let PortDecls::Struct(fields_typ) = struct_typ else {
            panic!("Input of `proj` should have struct value");
        };

        fields_typ
            .into_iter()
            .enumerate()
            .map(|(index, (name, _))| (name, Expr::Member { inner: input, index }.into()))
            .collect()
    }

    /// Unprojection.
    pub fn unproj(inner: Vec<(Option<String>, ExprId)>) -> ExprId {
        Expr::Struct { inner }.into()
    }

    /// Checks whether two exprs are equal.
    pub fn is_eq(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::EqArithmetic, lhs, rhs }.into()
    }

    /// Checks whether `lhs` is less than `rhs`.
    pub fn is_lt(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::Less, lhs, rhs }.into()
    }

    /// Checks whether `lhs` is greater than `rhs`.
    pub fn is_gt(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::Greater, lhs, rhs }.into()
    }

    /// Checks whether `lhs` is less or equal than `rhs`.
    pub fn is_le(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::LessEq, lhs, rhs }.into()
    }

    /// Checks whether `lhs` is greater or equal than `rhs`.
    pub fn is_ge(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::GreaterEq, lhs, rhs }.into()
    }

    /// `lhs` + `rhs`.
    pub fn add(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::Add, lhs, rhs }.into()
    }

    /// `lhs` - `rhs`.
    pub fn sub(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::Sub, lhs, rhs }.into()
    }

    /// `lhs` | `rhs`.
    pub fn bitor(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::Or, lhs, rhs }.into()
    }

    /// `lhs` & `rhs`.
    pub fn bitand(lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::BinaryOp { op: BinaryOp::And, lhs, rhs }.into()
    }

    /// Cond operator.
    pub fn cond(cond: ExprId, lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::Cond { cond, lhs, rhs }.into()
    }

    /// Not operator.
    pub fn not(inner: ExprId) -> ExprId {
        Expr::Not { inner }.into()
    }

    /// Clips the array `inner` (w/ type `[elt_typ; n]`) by `[from+:count]`.
    pub fn clip(elt_typ: PortDecls, n: usize, inner: ExprId, from: usize, count: usize) -> ExprId {
        let mut bits = usize_to_bitvec(n, from).to_vec();
        bits.truncate(clog2(n));

        Expr::Clip {
            inner,
            from: Expr::Constant { bits, typ: PortDecls::uint(clog2(n)) }.into(),
            size: count,
            typ_elt: elt_typ,
        }
        .into()
    }

    /// Appends exprs.
    pub fn append(elt_typ: PortDecls, lhs: ExprId, rhs: ExprId) -> ExprId {
        Expr::Append { lhs, rhs, typ_elt: elt_typ }.into()
    }

    /// Resizes by truncation or zero-extension
    pub fn resize(elt_typ: PortDecls, inner: ExprId, n: usize, m: usize) -> ExprId {
        match n.cmp(&m) {
            Ordering::Less => {
                let elt = Expr::Constant { bits: vec![false; elt_typ.width()], typ: elt_typ.clone() }.into();
                let elts = Expr::Repeat { inner: elt, count: m - n }.into();
                Expr::append(elt_typ, inner, elts)
            }
            Ordering::Equal => inner,
            Ordering::Greater => Expr::clip(elt_typ, n, inner, 0, m),
        }
    }

    /// Constructs expr from `usize` value.
    pub fn from_usize(value: usize, width: usize) -> ExprId {
        Expr::Constant { bits: usize_to_bitvec(width, value), typ: PortDecls::uint(width) }.into()
    }

    /// Get `index`-th element.
    pub fn get(elt_typ: PortDecls, inner: ExprId, index: ExprId) -> ExprId {
        Expr::Get { inner, typ_elt: elt_typ, index }.into()
    }

    /// Set `index`-th element to `elt`
    pub fn set(inner: ExprId, index: ExprId, elt: ExprId) -> ExprId {
        Expr::Set { inner, index, elt }.into()
    }

    /// Repeats expr.
    pub fn repeat(inner: ExprId, count: usize) -> ExprId {
        Expr::Repeat { inner, count }.into()
    }
}
