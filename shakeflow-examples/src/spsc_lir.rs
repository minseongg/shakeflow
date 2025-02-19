//! FIFO implemented in LIR.

use shakeflow::*;

fn spsc(k: &mut lir::CompositeModule, interface_typ: lir::InterfaceTyp) -> (lir::Module, lir::Module) {
    let interface_typ = lir::InterfaceTyp::Struct(
        [("0".to_string(), (None, interface_typ)), ("1".to_string(), (None, lir::InterfaceTyp::Unit))]
            .into_iter()
            .collect(),
    );

    let module = lir::composite("channel", interface_typ, Some("in"), Some("out"), |i, _| {
        let lir::Interface::Struct(mut inner) = i else {
            panic!();
        };

        let i = inner.remove("0").unwrap().1;

        lir::Interface::Struct(
            [("0".to_string(), (None, lir::Interface::Unit)), ("1".to_string(), (None, i))].into_iter().collect(),
        )
    })
    .build();

    k.register_inline(module).split()
}

/// SPSC channel.
pub fn channel(interface_typ: lir::InterfaceTyp) {
    let module = lir::composite("spsc_lir", interface_typ.clone(), Some("in"), Some("out"), |value, k| {
        let (sink, source) = spsc(k, interface_typ);

        let ret = lir::Interface::Unit.comb_inline(k, source);
        value.comb_inline(k, sink);
        ret
    })
    .build();

    let module: vir::Module =
        codegen::gen_module::<Virgen>(String::from("fifo_lir"), &module).expect("Failed to codegen").into();

    println!("{}", module);
}
