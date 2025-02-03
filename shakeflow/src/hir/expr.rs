use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::*;

use arrayvec::ArrayVec;
use hashcons::merkle::Merkle;

use crate::hir::*;
use crate::lir;
use crate::utils::{clog2, usize_to_bitvec};

/// Store the expr in the heap storage.
fn store_expr<'a, V: Signal>(table: &lir::Table, expr: Expr<V>) -> &'a Expr<V> {
    let expr_ptr = Box::into_raw(Box::new(expr));

    // SAFETY: It is safe to call `Box::from_raw` because created `expr_ptr` is different for each method call.
    let expr = unsafe { Box::from_raw(expr_ptr as *mut usize as *mut Expr<V>) };
    table.storage.borrow_mut().push(expr);

    // SAFETY: It is safe to dereference the raw pointer because the storage is dropped after the target code generation.
    unsafe { &*expr_ptr }
}

impl<V: Signal> lir::TableStorageElement for Expr<V> {}

/// Exprs.
#[derive(Debug, Clone)]
pub struct Expr<V: Signal> {
    /// Inner expr id.
    id: lir::ExprId,

    _marker: PhantomData<V>,
}

impl<V: Signal> Copy for Expr<V> {}

impl<V: Signal> Expr<V> {
    /// Don't care value.
    pub fn x() -> Self {
        lir::Expr::x(V::port_decls()).into()
    }

    /// Input expr.
    pub fn input(name: Option<String>) -> Self {
        lir::Expr::input(V::port_decls(), name).into()
    }

    /// Member of input expr.
    pub fn member<W: Signal>(input: Expr<W>, index: usize) -> Self {
        match W::port_decls() {
            lir::PortDecls::Struct(inner) => {
                assert_eq!(V::port_decls(), inner[index].1);
            }
            _ => panic!("Input of `member` should have struct value"),
        }
        lir::Expr::member(W::port_decls(), input.id, index).into()
    }

    /// Consumes the `Expr`, returning the wrapped `lir::ExprId`.
    pub fn into_inner(self) -> lir::ExprId {
        self.id
    }

    /// Case expression
    pub fn case<W1: Signal, W2: Signal>(
        &self, case_items: Vec<(Expr<W1>, Expr<W2>)>, default: Option<Expr<W2>>,
    ) -> Expr<W2> {
        lir::Expr::Case {
            case_expr: self.into_inner(),
            case_items: case_items.iter().map(|(x, y)| (x.into_inner(), y.into_inner())).collect::<Vec<_>>(),
            default: default.map(|d| d.into_inner()),
        }
        .into()
    }
}

impl<V: Signal, const N: usize> FromIterator<Expr<V>> for [Expr<V>; N] {
    fn from_iter<T: IntoIterator<Item = Expr<V>>>(iter: T) -> Self {
        iter.into_iter().collect::<ArrayVec<Expr<V>, N>>().into_inner().unwrap()
    }
}

impl<V: Signal> From<lir::Expr> for Expr<V> {
    /// Constructs expr from LIR expr.
    fn from(inner: lir::Expr) -> Self {
        assert_eq!(V::port_decls(), inner.port_decls());
        Self { id: lir::ExprId::from(inner), _marker: PhantomData }
    }
}

impl<V: Signal> From<lir::ExprId> for Expr<V> {
    fn from(id: lir::ExprId) -> Self {
        Self { id, _marker: PhantomData }
    }
}

impl<V: Signal> From<V> for Expr<V> {
    /// Synthesizes expr from constant.
    fn from(signal: V) -> Self {
        assert_eq!(V::port_decls().max_dim(), 1);
        signal.to_lir().into()
    }
}

impl<N: Num> From<usize> for Expr<Bits<N>> {
    fn from(signal: usize) -> Self {
        Bits::new(usize_to_bitvec(N::WIDTH, signal)).into()
    }
}

impl<const N: usize> From<[bool; N]> for Expr<Bits<U<N>>> {
    fn from(signal: [bool; N]) -> Self {
        Self::from(Bits::new(signal.into_iter().collect()))
    }
}

impl<V: Signal> From<Expr<Bits<U<{ V::WIDTH }>>>> for Expr<V> {
    fn from(expr: Expr<Bits<U<{ V::WIDTH }>>>) -> Self {
        assert_eq!(V::port_decls().max_dim(), 1);
        Self { id: expr.id, _marker: PhantomData }
    }
}

impl<V: Signal> From<Expr<V>> for Expr<Bits<U<{ V::WIDTH }>>> {
    fn from(expr: Expr<V>) -> Self {
        expr.repr()
    }
}

impl<V: Signal, const N: usize> From<[Expr<V>; N]> for Expr<Array<V, U<N>>> {
    // FIXME: currently this typecast generates verilog with unnecesary amount of register
    // declaration, since each set operation allocates register. This needs to be fixed by creating
    // a new lir::Expr and implementing codegen for this specific case.
    fn from(expr: [Expr<V>; N]) -> Self {
        lir::Expr::ConcatArray {
            inner: expr.into_iter().map(|expr| expr.into_inner()).collect::<Vec<_>>(),
            elt_typ: V::port_decls(),
        }
        .into()
    }
}

impl<V: Signal, N: Num> Expr<Array<V, N>> {
    /// Set `index`-th element to `elt`
    pub fn set(&self, index: Expr<Bits<Log2<N>>>, elt: Expr<V>) -> Expr<Array<V, N>> {
        assert_eq!(V::port_decls().max_dim(), 1);
        lir::Expr::Set { inner: self.into_inner(), index: index.into_inner(), elt: elt.into_inner() }.into()
    }

    /// Set M elements from `index` to `elts`
    pub fn set_range<M: Num>(&self, index: Expr<Bits<Log2<N>>>, elts: Expr<Array<V, M>>) -> Expr<Array<V, N>> {
        assert_eq!(V::port_decls().max_dim(), 1);

        lir::Expr::SetRange {
            inner: self.into_inner(),
            typ_elt: V::port_decls(),
            index: index.into_inner(),
            elts: elts.into_inner(),
        }
        .into()
    }

    /// Fold Left
    pub fn fold<W: Signal>(&self, init: Expr<W>, f: fn(Expr<W>, Expr<V>) -> Expr<W>) -> Expr<W> {
        let acc: Expr<W> = Expr::input(Some("acc".to_string()));
        let inner_slice: Expr<V> = Expr::input(Some("inner_slice".to_string()));
        let output: Expr<W> = f(acc, inner_slice);

        lir::Expr::Fold {
            inner: self.into_inner(),
            typ_elt: V::port_decls(),
            func: output.into_inner(),
            init: init.into_inner(),
            acc: acc.into_inner(),
            inner_slice: inner_slice.into_inner(),
        }
        .into()
    }

    /// Tree fold
    /// This operation folds an array with 2^K elements by constructing a fold tree(with height K) as below:
    /// ```text
    ///   O   O    ...  O   O
    ///    \ / (op)      \ / (op)
    ///     O     ...     O
    ///
    ///           ...
    ///
    ///           \/
    ///           O
    /// ````
    ///
    /// This operation can generated better verilog, but need to be used carefully
    ///
    /// 1. Associativity of the operation
    ///
    /// Unlike the `hir::fold`, which is foldleft, the order of operation will rearranged
    /// arbitrarily. So if the operation is not associative, the result might be different from
    /// expected.
    ///
    /// 2. Number of elements
    ///
    /// In order to construct the fold tree in a readable way in verilog (which is nested for loop),
    /// we only allow use of this api only when length is power of 2 (ex. 1, 2, 4, 8, ...).
    /// You should manually resize to use this api for arrays that does not satisfy the constraint
    pub fn tree_fold(&self, op: fn(Expr<V>, Expr<V>) -> Expr<V>) -> Expr<V> {
        // TODO: more static way to check number of elt is 2^N
        assert!(N::WIDTH.is_power_of_two());

        let lhs: Expr<V> = Expr::input(Some("lhs".to_string()));
        let rhs: Expr<V> = Expr::input(Some("rhs".to_string()));
        let acc: Self = Expr::input(Some("acc".to_string()));
        let op = op(lhs, rhs);

        lir::Expr::TreeFold {
            inner: self.into_inner(),
            op: op.into_inner(),
            acc: acc.into_inner(),
            lhs: lhs.into_inner(),
            rhs: rhs.into_inner(),
        }
        .into()
    }

    /// Maps each element expr.
    pub fn map<W: Signal>(&self, f: fn(Expr<V>) -> Expr<W>) -> Expr<Array<W, N>> {
        // TODO: Give proper name.
        assert_eq!(V::port_decls().max_dim(), 1);
        assert_eq!(W::port_decls().max_dim(), 1);
        let input = Expr::input(None);
        let output = f(input);

        lir::Expr::Map { inner: self.into_inner(), typ_elt: V::port_decls(), func: output.into_inner() }.into()
    }

    /// Clips the array (from: Expr)
    pub fn clip<SZ: Num>(&self, from: Expr<Bits<Log2<N>>>) -> Expr<Array<V, SZ>> {
        assert_eq!(V::port_decls().max_dim(), 1);
        lir::Expr::Clip { inner: self.into_inner(), from: from.into_inner(), size: SZ::WIDTH, typ_elt: V::port_decls() }
            .into()
    }

    /// Clips the array.
    pub fn clip_const<SZ: Num>(&self, from: usize) -> Expr<Array<V, SZ>> {
        assert_eq!(V::port_decls().max_dim(), 1);
        assert!(0 < SZ::WIDTH && from + SZ::WIDTH - 1 < N::WIDTH);
        let mut bits = usize_to_bitvec(N::WIDTH, from).to_vec();
        bits.truncate(clog2(N::WIDTH));
        lir::Expr::Clip {
            inner: self.into_inner(),
            from: lir::ExprId::alloc_expr(Merkle::new(lir::Expr::Constant {
                bits,
                typ: lir::PortDecls::Bits(lir::Shape::new([clog2(N::WIDTH)])),
            })),
            size: SZ::WIDTH,
            typ_elt: V::port_decls(),
        }
        .into()
    }

    /// Appends exprs.
    /// `a.append(b)` corresponds to `{b, a}` of Verilog code.
    pub fn append<M: Num>(&self, rhs: Expr<Array<V, M>>) -> Expr<Array<V, Sum<N, M>>> {
        assert_eq!(V::port_decls().max_dim(), 1);
        lir::Expr::Append { lhs: self.into_inner(), rhs: rhs.into_inner(), typ_elt: V::port_decls() }.into()
    }

    /// Resize by truncation or zero-extension
    pub fn resize<M: Num>(&self) -> Expr<Array<V, M>> {
        assert_eq!(V::port_decls().max_dim(), 1);

        match N::WIDTH.cmp(&M::WIDTH) {
            Ordering::Less => lir::Expr::Append {
                lhs: self.into_inner(),
                rhs: lir::ExprId::alloc_expr(Merkle::new(lir::Expr::Repeat {
                    inner: lir::ExprId::alloc_expr(Merkle::new(lir::Expr::Constant {
                        bits: vec![false; V::WIDTH],
                        typ: V::port_decls(),
                    })),
                    count: M::WIDTH - N::WIDTH,
                })),
                typ_elt: V::port_decls(),
            }
            .into(),
            Ordering::Equal => Expr { id: self.id, _marker: PhantomData },
            Ordering::Greater => self.clip_const::<M>(0),
        }
    }
}

impl<V1: Signal, N: Num> Expr<Array<V1, N>> {
    /// Zips with another array expr.
    pub fn zip<V2: Signal>(&self, other: Expr<Array<V2, N>>) -> Expr<Array<(V1, V2), N>> {
        assert_eq!(V1::port_decls().max_dim(), 1);
        assert_eq!(V2::port_decls().max_dim(), 1);

        lir::Expr::Zip {
            inner: vec![self.into_inner(), other.into_inner()],
            typ_inner: vec![V1::port_decls(), V2::port_decls()],
        }
        .into()
    }

    /// Zips three exprs.
    pub fn zip3<V2: Signal, V3: Signal>(
        &self, other1: Expr<Array<V2, N>>, other2: Expr<Array<V3, N>>,
    ) -> Expr<Array<(V1, V2, V3), N>> {
        assert_eq!(V1::port_decls().max_dim(), 1);
        assert_eq!(V2::port_decls().max_dim(), 1);
        assert_eq!(V3::port_decls().max_dim(), 1);

        lir::Expr::Zip {
            inner: vec![self.into_inner(), other1.into_inner(), other2.into_inner()],
            typ_inner: vec![V1::port_decls(), V2::port_decls(), V3::port_decls()],
        }
        .into()
    }

    /// Zips four exprs.
    pub fn zip4<V2: Signal, V3: Signal, V4: Signal>(
        &self, other1: Expr<Array<V2, N>>, other2: Expr<Array<V3, N>>, other3: Expr<Array<V4, N>>,
    ) -> Expr<Array<(V1, V2, V3, V4), N>> {
        assert_eq!(V1::port_decls().max_dim(), 1);
        assert_eq!(V2::port_decls().max_dim(), 1);
        assert_eq!(V3::port_decls().max_dim(), 1);
        assert_eq!(V4::port_decls().max_dim(), 1);

        lir::Expr::Zip {
            inner: vec![self.into_inner(), other1.into_inner(), other2.into_inner(), other3.into_inner()],
            typ_inner: vec![V1::port_decls(), V2::port_decls(), V3::port_decls(), V4::port_decls()],
        }
        .into()
    }
}

impl<V: Signal, const N: usize> Expr<Array<V, U<N>>> {
    /// Enumerate
    pub fn enumerate<M: Num>(&self) -> Expr<Array<(Bits<M>, V), U<N>>> {
        let range = range::<N, M>();
        range.zip(*self)
    }
}

impl<V: Signal, N: Num> Index<usize> for Expr<Array<V, N>> {
    type Output = Expr<V>;

    fn index(&self, index: usize) -> &Self::Output {
        assert_eq!(V::port_decls().max_dim(), 1);
        assert!(index < N::WIDTH);

        let mut bits = usize_to_bitvec(N::WIDTH, index).to_vec();
        bits.truncate(clog2(N::WIDTH));

        let expr = Expr::<V>::from(lir::Expr::Get {
            inner: self.into_inner(),
            typ_elt: V::port_decls(),
            index: lir::ExprId::alloc_expr(Merkle::new(lir::Expr::Constant {
                bits,
                typ: lir::PortDecls::Bits(lir::Shape::new([clog2(N::WIDTH)])),
            })),
        });

        lir::TABLE.with(|table| store_expr(table, expr))
    }
}

impl<V: Signal, N: Num> Index<Expr<Bits<Log2<N>>>> for Expr<Array<V, N>> {
    type Output = Expr<V>;

    fn index(&self, index: Expr<Bits<Log2<N>>>) -> &Self::Output {
        assert_eq!(V::port_decls().max_dim(), 1);

        let expr = Expr::<V>::from(lir::Expr::Get {
            inner: self.into_inner(),
            typ_elt: V::port_decls(),
            index: index.into_inner(),
        });

        lir::TABLE.with(|table| store_expr(table, expr))
    }
}

impl<V: Signal, M: Num, N: Num> Expr<Array<Array<V, M>, N>> {
    /// Concatenates array elements.
    pub fn concat(&self) -> Expr<Array<V, Prod<N, M>>> {
        assert_eq!(V::port_decls().max_dim(), 1);
        lir::Expr::Concat { inner: self.into_inner(), typ_elt: V::port_decls() }.into()
    }
}

impl<V: Signal, N: Num> Expr<Array<V, N>> {
    /// Splits array into chunks.
    ///
    /// `chunk` can be used to implement adder tree. Obtaining sum of two adjacent nodes in the adder tree
    /// can be implemented by splitting them into chunks of size 2 and then obtaining the sum of each chunk.
    pub fn chunk<M: Num>(&self) -> Expr<Array<Array<V, M>, Quot<N, M>>> {
        assert_eq!(V::port_decls().max_dim(), 1);
        lir::Expr::Chunk { inner: self.into_inner(), chunk_size: M::WIDTH }.into()
    }
}

/// Returns range (0..N).
pub fn range<const N: usize, M: Num>() -> Expr<Array<Bits<M>, U<N>>> {
    let range: [Expr<Bits<M>>; N] = (0..N).map(Expr::<Bits<M>>::from).collect::<Vec<_>>().try_into().unwrap();
    range.into()
}

impl<M: Num, N: Num> Expr<Array<Bits<M>, N>> {
    /// Sums the values.
    /// Note: Ignores carry. TODO: Include carry.
    ///
    /// Note: Although having the `lir::Expr` enum have a `Sum` member is API-wise unintuitive
    ///       (as `.sum()` is often considered to be a hir-level operation instead of lir-level),
    ///       we set it regardless because it needs to be parsed at codegen-level to create an
    ///       actual for-loop instead of a manually unrolled loop.
    pub fn sum(&self) -> Expr<Bits<M>> {
        lir::Expr::Sum { inner: self.into_inner(), width_elt: M::WIDTH }.into()
    }
}

impl<V: Signal> Expr<V> {
    /// Converts to bit representation.
    pub fn repr(&self) -> Expr<Bits<U<{ V::WIDTH }>>> {
        assert_eq!(V::port_decls().max_dim(), 1);
        lir::Expr::Repr { inner: self.into_inner() }.into()
    }

    /// Repeats expr.
    pub fn repeat<N: Num>(&self) -> Expr<Array<V, N>> {
        assert_eq!(V::port_decls().max_dim(), 1);
        lir::Expr::Repeat { inner: self.into_inner(), count: N::WIDTH }.into()
    }
}

impl<N: Num> Expr<Bits<N>> {
    /// $signed() system function
    pub fn signed(&self) -> Self {
        lir::Expr::Call {
            func_name: "$signed".to_string(),
            args: vec![self.into_inner()],
            typ: <Bits<N> as Signal>::port_decls(),
        }
        .into()
    }
}

impl<N: Num> Add<Expr<Bits<N>>> for Expr<Bits<N>> {
    type Output = Expr<Bits<Sum<N, U<1>>>>;

    fn add(self, rhs: Expr<Bits<N>>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Add, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> Sub<Expr<Bits<N>>> for Expr<Bits<N>> {
    type Output = Expr<Bits<N>>;

    fn sub(self, rhs: Expr<Bits<N>>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Sub, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num, M: Num> Mul<Expr<Bits<M>>> for Expr<Bits<N>> {
    type Output = Expr<Bits<Sum<N, M>>>;

    fn mul(self, rhs: Expr<Bits<M>>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Mul, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num, M: Num> Div<Expr<Bits<M>>> for Expr<Bits<N>> {
    type Output = Expr<Bits<N>>;

    fn div(self, rhs: Expr<Bits<M>>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Div, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num, M: Num> Rem<Expr<Bits<M>>> for Expr<Bits<N>> {
    type Output = Expr<Bits<M>>;

    fn rem(self, rhs: Expr<Bits<M>>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Mod, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl BitOr<Self> for Expr<bool> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Or, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl BitAnd<Self> for Expr<bool> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::And, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl BitXor<Self> for Expr<bool> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Xor, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitOr<Expr<Bits<N>>> for Expr<bool> {
    type Output = Expr<Bits<N>>;

    fn bitor(self, rhs: Expr<Bits<N>>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Or, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitAnd<Expr<Bits<N>>> for Expr<bool> {
    type Output = Expr<Bits<N>>;

    fn bitand(self, rhs: Expr<Bits<N>>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::And, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitXor<Expr<Bits<N>>> for Expr<bool> {
    type Output = Expr<Bits<N>>;

    fn bitxor(self, rhs: Expr<Bits<N>>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Xor, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitOr<Self> for Expr<Bits<N>> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Or, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitAnd<Self> for Expr<Bits<N>> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::And, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitXor<Self> for Expr<Bits<N>> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Xor, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitOr<Expr<bool>> for Expr<Bits<N>> {
    type Output = Self;

    fn bitor(self, rhs: Expr<bool>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Or, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitAnd<Expr<bool>> for Expr<Bits<N>> {
    type Output = Self;

    fn bitand(self, rhs: Expr<bool>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::And, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> BitXor<Expr<bool>> for Expr<Bits<N>> {
    type Output = Self;

    fn bitxor(self, rhs: Expr<bool>) -> Self::Output {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Xor, lhs: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> Shl<usize> for Expr<Bits<N>> {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        let mut bits = usize_to_bitvec(N::WIDTH, rhs);
        bits.truncate(clog2(N::WIDTH));
        lir::Expr::LeftShift {
            inner: self.into_inner(),
            rhs: lir::ExprId::alloc_expr(Merkle::new(lir::Expr::Constant {
                bits,
                typ: lir::PortDecls::Bits(lir::Shape::new([clog2(N::WIDTH)])),
            })),
        }
        .into()
    }
}

impl<const N: usize> Shl<Expr<Bits<U<{ clog2(N) }>>>> for Expr<Bits<U<N>>> {
    type Output = Self;

    fn shl(self, rhs: Expr<Bits<U<{ clog2(N) }>>>) -> Self::Output {
        lir::Expr::LeftShift { inner: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> Shl<Expr<Bits<Log2<N>>>> for Expr<Bits<N>> {
    type Output = Self;

    fn shl(self, rhs: Expr<Bits<Log2<N>>>) -> Self::Output {
        lir::Expr::LeftShift { inner: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<N: Num> Shr<usize> for Expr<Bits<N>> {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        let mut bits = usize_to_bitvec(N::WIDTH, rhs);
        bits.truncate(clog2(N::WIDTH));
        lir::Expr::RightShift {
            inner: self.into_inner(),
            rhs: lir::ExprId::alloc_expr(Merkle::new(lir::Expr::Constant {
                bits,
                typ: lir::PortDecls::Bits(lir::Shape::new([clog2(N::WIDTH)])),
            })),
        }
        .into()
    }
}

impl<N: Num> Shr<Expr<Bits<Log2<N>>>> for Expr<Bits<N>> {
    type Output = Self;

    fn shr(self, rhs: Expr<Bits<Log2<N>>>) -> Self::Output {
        lir::Expr::RightShift { inner: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<const N: usize> Shr<Expr<Bits<U<{ clog2(N) }>>>> for Expr<Bits<U<N>>> {
    type Output = Self;

    fn shr(self, rhs: Expr<Bits<U<{ clog2(N) }>>>) -> Self::Output {
        lir::Expr::RightShift { inner: self.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

/// Trait for derived enum values
/// Every derived enum Values implement this
pub trait EnumValue: Signal {}

impl<V> Expr<V>
where
    V: EnumValue,
    [(); V::WIDTH]:,
{
    /// Check two enums are equal.
    pub fn is_eq(&self, other: Expr<V>) -> Expr<bool> {
        // FIXME: use repr for now, due to compiler error
        let lhs = (*self).repr();
        let rhs = other.repr();
        lhs.is_eq(rhs)
    }
}

impl<N: Num> Expr<Bits<N>> {
    /// Check two exprs are equal.
    pub fn is_eq(&self, other: Expr<Bits<N>>) -> Expr<bool> {
        lir::Expr::BinaryOp { op: lir::BinaryOp::EqArithmetic, lhs: self.into_inner(), rhs: other.into_inner() }.into()
    }

    /// Check `self` is less than `other`.
    pub fn is_lt(&self, other: Expr<Bits<N>>) -> Expr<bool> {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Less, lhs: self.into_inner(), rhs: other.into_inner() }.into()
    }

    /// Check `self` is greater than `other`.
    pub fn is_gt(&self, other: Expr<Bits<N>>) -> Expr<bool> {
        lir::Expr::BinaryOp { op: lir::BinaryOp::Greater, lhs: self.into_inner(), rhs: other.into_inner() }.into()
    }

    /// Check `self` is less or equal than `other`.
    pub fn is_le(&self, other: Expr<Bits<N>>) -> Expr<bool> {
        lir::Expr::BinaryOp { op: lir::BinaryOp::LessEq, lhs: self.into_inner(), rhs: other.into_inner() }.into()
    }

    /// Check `self` is greater or equal than `other`.
    pub fn is_ge(&self, other: Expr<Bits<N>>) -> Expr<bool> {
        lir::Expr::BinaryOp { op: lir::BinaryOp::GreaterEq, lhs: self.into_inner(), rhs: other.into_inner() }.into()
    }

    /// Check any exprs are asserted
    pub fn any(&self) -> Expr<bool> {
        let acc: Expr<bool> = Expr::input(Some("acc".to_string()));
        let inner_slice: Expr<bool> = Expr::input(Some("inner_slice".to_string()));
        let f = |acc, inner_slice| acc | inner_slice;

        lir::Expr::Fold {
            inner: self.into_inner(),
            typ_elt: bool::port_decls(),
            func: f(acc, inner_slice).into_inner(),
            init: Expr::from(false).into_inner(),
            acc: acc.into_inner(),
            inner_slice: inner_slice.into_inner(),
        }
        .into()
    }

    /// Check all exprs are asserted
    pub fn all(&self) -> Expr<bool> {
        let acc: Expr<bool> = Expr::input(Some("acc".to_string()));
        let inner_slice: Expr<bool> = Expr::input(Some("inner_slice".to_string()));
        let f = |acc, inner_slice| acc & inner_slice;

        lir::Expr::Fold {
            inner: self.into_inner(),
            typ_elt: bool::port_decls(),
            func: f(acc, inner_slice).into_inner(),
            init: Expr::from(true).into_inner(),
            acc: acc.into_inner(),
            inner_slice: inner_slice.into_inner(),
        }
        .into()
    }
}

impl<const N: usize> Expr<Bits<U<N>>> {
    /// This method accepts byte string composed of only [0, 1, ?] and returns whether expression
    /// matches given byte string.
    pub fn is_bitpat(&self, pat: &[u8; N]) -> Expr<bool> {
        let mask = Bits::new(pat.iter().rev().map(|c| c != &b'?').collect::<Vec<_>>());
        let result = Bits::new(pat.iter().rev().map(|c| c == &b'1').collect::<Vec<_>>());
        (*self & Expr::from(mask)).is_eq(Expr::from(result))
    }
}

impl Not for Expr<bool> {
    type Output = Self;

    fn not(self) -> Self::Output {
        lir::Expr::Not { inner: self.into_inner() }.into()
    }
}

impl<V: Signal, N: Num> Not for Expr<Array<V, N>>
where Expr<V>: Not
{
    type Output = Self;

    fn not(self) -> Self::Output {
        lir::Expr::Not { inner: self.into_inner() }.into()
    }
}

impl Expr<bool> {
    /// Cond operator (aka. if-then-else).
    pub fn cond<V: Signal>(&self, lhs: Expr<V>, rhs: Expr<V>) -> Expr<V> {
        lir::Expr::Cond { cond: self.into_inner(), lhs: lhs.into_inner(), rhs: rhs.into_inner() }.into()
    }
}

impl<V: Signal, N: Num> Index<Expr<Bits<Log2<N>>>> for Expr<VarArray<V, N>> {
    type Output = Expr<V>;

    fn index(&self, index: Expr<Bits<Log2<N>>>) -> &Self::Output {
        let expr = Expr::<V>::from(lir::Expr::GetVarArray {
            inner: self.into_inner(),
            typ_elt: V::port_decls(),
            index: index.into_inner(),
        });

        lir::TABLE.with(|table| store_expr(table, expr))
    }
}

impl<V: Signal, N: Num> Expr<VarArray<V, N>> {
    /// TODO: Documentation
    pub fn set_var_arr(&self, index: Expr<Bits<Log2<N>>>, elt: Expr<V>) -> Self {
        lir::Expr::SetVarArray { inner: self.into_inner(), index: index.into_inner(), elt: elt.into_inner() }.into()
    }
}

/// Serializes into bit array.
///
/// TODO: Use `repr` method instead of this trait
pub trait Serialize: Signal {
    /// Serialization.
    fn serialize(expr: Expr<Self>) -> Expr<Bits<U<{ Self::WIDTH }>>>;
}

impl<V: Serialize> Expr<V> {
    /// Serialization.
    pub fn serialize(&self) -> Expr<Bits<U<{ V::WIDTH }>>> {
        Serialize::serialize(*self)
    }
}

/// Deserializes from bit array.
///
/// TODO: Use `from` method instead of this trait
pub trait Deserialize: Signal {
    /// Deserialize.
    fn deserialize(expr: Expr<Bits<U<{ Self::WIDTH }>>>) -> Expr<Self>;
}

/// Trait for expr projection.
pub trait ExprProj: Signal {
    /// Projected type.
    type Target: lir::TableStorageElement;

    /// Projection.
    fn proj(expr: Expr<Self>) -> Self::Target;
}

impl<V: ExprProj> Deref for Expr<V> {
    type Target = V::Target;

    fn deref(&self) -> &Self::Target {
        let expr_ptr = Box::into_raw(Box::new(ExprProj::proj(*self)));

        // # Safety
        //
        // It is safe to call `Box::from_raw` because created `expr_ptr` is different for each method call.
        let expr = unsafe { Box::from_raw(expr_ptr as *mut usize as *mut V::Target) };
        lir::TABLE.with(|table| table.storage.borrow_mut().push(expr));

        // # Safety
        //
        // It is safe to dereference the raw pointer because the storage is dropped after the target code generation.
        unsafe { &*expr_ptr }
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! impl_table_storage_element_tuple {
    ($($a:ident)+) => {
        impl<$($a: Signal,)+> lir::TableStorageElement for ($(Expr<$a>,)+) {}
    }
}

impl_table_storage_element_tuple! { V1 V2 }
impl_table_storage_element_tuple! { V1 V2 V3 }
impl_table_storage_element_tuple! { V1 V2 V3 V4 }
impl_table_storage_element_tuple! { V1 V2 V3 V4 V5 }
impl_table_storage_element_tuple! { V1 V2 V3 V4 V5 V6 }

impl<A: Signal, B: Signal> ExprProj for (A, B) {
    type Target = (Expr<A>, Expr<B>);

    /// Projects a pair expr to expr pair.
    fn proj(expr: Expr<Self>) -> Self::Target {
        (Expr::member(expr, 0), Expr::member(expr, 1))
    }
}

impl<A: Signal, B: Signal, C: Signal> ExprProj for (A, B, C) {
    type Target = (Expr<A>, Expr<B>, Expr<C>);

    /// Projects a pair expr to expr pair.
    fn proj(expr: Expr<Self>) -> Self::Target {
        (Expr::member(expr, 0), Expr::member(expr, 1), Expr::member(expr, 2))
    }
}

impl<A: Signal, B: Signal, C: Signal, D: Signal> ExprProj for (A, B, C, D) {
    type Target = (Expr<A>, Expr<B>, Expr<C>, Expr<D>);

    /// Projects a pair expr to expr pair.
    fn proj(expr: Expr<Self>) -> Self::Target {
        (Expr::member(expr, 0), Expr::member(expr, 1), Expr::member(expr, 2), Expr::member(expr, 3))
    }
}

impl<A: Signal, B: Signal, C: Signal, D: Signal, E: Signal> ExprProj for (A, B, C, D, E) {
    type Target = (Expr<A>, Expr<B>, Expr<C>, Expr<D>, Expr<E>);

    /// Projects a pair expr to expr pair.
    fn proj(expr: Expr<Self>) -> Self::Target {
        (
            Expr::member(expr, 0),
            Expr::member(expr, 1),
            Expr::member(expr, 2),
            Expr::member(expr, 3),
            Expr::member(expr, 4),
        )
    }
}

#[allow(clippy::type_complexity)]
impl<A: Signal, B: Signal, C: Signal, D: Signal, E: Signal, F: Signal> ExprProj for (A, B, C, D, E, F) {
    type Target = (Expr<A>, Expr<B>, Expr<C>, Expr<D>, Expr<E>, Expr<F>);

    /// Projects a pair expr to expr pair.
    fn proj(expr: Expr<Self>) -> Self::Target {
        (
            Expr::member(expr, 0),
            Expr::member(expr, 1),
            Expr::member(expr, 2),
            Expr::member(expr, 3),
            Expr::member(expr, 4),
            Expr::member(expr, 5),
        )
    }
}

// TODO: maybe `unproj()` should be a member of `ExprProj` to generically handle `from_pair*()`.
impl<V1: Signal, V2: Signal> From<(Expr<V1>, Expr<V2>)> for Expr<(V1, V2)> {
    /// Unprojects a pair expr to expr pair.
    fn from((s1, s2): (Expr<V1>, Expr<V2>)) -> Self {
        lir::Expr::Struct {
            inner: vec![(Some("0".to_string()), s1.into_inner()), (Some("1".to_string()), s2.into_inner())],
        }
        .into()
    }
}

impl<V1: Signal, V2: Signal, V3: Signal> From<(Expr<V1>, Expr<V2>, Expr<V3>)> for Expr<(V1, V2, V3)> {
    /// Unprojects a pair expr to expr pair.
    fn from((s1, s2, s3): (Expr<V1>, Expr<V2>, Expr<V3>)) -> Self {
        lir::Expr::Struct {
            inner: vec![
                (Some("0".to_string()), s1.into_inner()),
                (Some("1".to_string()), s2.into_inner()),
                (Some("2".to_string()), s3.into_inner()),
            ],
        }
        .into()
    }
}

impl<V1: Signal, V2: Signal, V3: Signal, V4: Signal> From<(Expr<V1>, Expr<V2>, Expr<V3>, Expr<V4>)>
    for Expr<(V1, V2, V3, V4)>
{
    /// Unprojects a pair expr to expr pair.
    fn from((s1, s2, s3, s4): (Expr<V1>, Expr<V2>, Expr<V3>, Expr<V4>)) -> Self {
        lir::Expr::Struct {
            inner: vec![
                (Some("0".to_string()), s1.into_inner()),
                (Some("1".to_string()), s2.into_inner()),
                (Some("2".to_string()), s3.into_inner()),
                (Some("3".to_string()), s4.into_inner()),
            ],
        }
        .into()
    }
}

impl<V1: Signal, V2: Signal, V3: Signal, V4: Signal, V5: Signal>
    From<(Expr<V1>, Expr<V2>, Expr<V3>, Expr<V4>, Expr<V5>)> for Expr<(V1, V2, V3, V4, V5)>
{
    /// Unprojects a pair expr to expr pair.
    fn from((s1, s2, s3, s4, s5): (Expr<V1>, Expr<V2>, Expr<V3>, Expr<V4>, Expr<V5>)) -> Self {
        lir::Expr::Struct {
            inner: vec![
                (Some("0".to_string()), s1.into_inner()),
                (Some("1".to_string()), s2.into_inner()),
                (Some("2".to_string()), s3.into_inner()),
                (Some("3".to_string()), s4.into_inner()),
                (Some("4".to_string()), s5.into_inner()),
            ],
        }
        .into()
    }
}

impl<V1: Signal, V2: Signal, V3: Signal, V4: Signal, V5: Signal, V6: Signal>
    From<(Expr<V1>, Expr<V2>, Expr<V3>, Expr<V4>, Expr<V5>, Expr<V6>)> for Expr<(V1, V2, V3, V4, V5, V6)>
{
    /// Unprojects a pair expr to expr pair.
    fn from((s1, s2, s3, s4, s5, s6): (Expr<V1>, Expr<V2>, Expr<V3>, Expr<V4>, Expr<V5>, Expr<V6>)) -> Self {
        lir::Expr::Struct {
            inner: vec![
                (Some("0".to_string()), s1.into_inner()),
                (Some("1".to_string()), s2.into_inner()),
                (Some("2".to_string()), s3.into_inner()),
                (Some("3".to_string()), s4.into_inner()),
                (Some("4".to_string()), s5.into_inner()),
                (Some("5".to_string()), s6.into_inner()),
            ],
        }
        .into()
    }
}

/// Selects from a set of exprs.
/// TODO: Add usage example
#[macro_export]
macro_rules! select {
    (
        default => $a:expr,
    ) => {
        $a
    };
    (
        $a:expr => $b:expr,
        $($c:tt)*
    ) => {
        $a.cond($b, select!($($c)*))
    }
}

/// If `cond` is true, then set the `idx`-th element of `id` to `elt`.
#[macro_export]
macro_rules! if_then_set {
    (
        $id:expr, $cond:expr, $idx:expr, $elt:expr
    ) => {
        $id.set($idx, $cond.cond($elt, $id[$idx]))
    };
}

/// If `cond` is true, then set the `idx`-th element of `id` to `elt`. (for VarArray)
#[macro_export]
macro_rules! if_then_set_var_arr {
    (
        $id:expr, $cond:expr, $idx:expr, $elt:expr
    ) => {
        $id.set_var_arr($idx, $cond.cond($elt, $id[$idx]))
    };
}
