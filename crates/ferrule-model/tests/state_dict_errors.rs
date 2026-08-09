use ferrule_common::{StateDictBindingIssue, StateDictSchemaError};
use ferrule_model::nn::{
    ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::transformer::{
    BindingIssue, ExactNameMapper, NameMapping, StateDictBinder, StateDictSchema,
};

fn parameter(id: u64, path: &str) -> ParameterSpec {
    ParameterSpec::new(
        ParameterId::new(id),
        ModulePath::new(path).unwrap(),
        ParameterDType::Bf16,
        [1],
        ParameterResidency::Static,
    )
    .unwrap()
}

#[test]
fn schema_and_binding_errors_use_common_typed_variants() {
    let first = parameter(1, "weight");
    let mut builder = StateDictSchema::builder();
    builder.register(first.clone()).unwrap();
    let duplicate = builder.register(first).unwrap_err();
    assert!(matches!(
        duplicate,
        StateDictSchemaError::DuplicatePath { .. }
    ));

    let mut schema = StateDictSchema::builder();
    schema.register(parameter(2, "other")).unwrap();
    let schema = schema.build().unwrap();
    let mut mapper = ExactNameMapper::new();
    mapper
        .insert(
            "external",
            NameMapping::weight(ModulePath::new("missing").unwrap()),
        )
        .unwrap();
    let error = StateDictBinder::new(&schema, &mapper)
        .bind_slices(Vec::new())
        .unwrap_err();
    assert!(error.issues().iter().any(|issue: &BindingIssue| matches!(
        issue,
        StateDictBindingIssue::MissingParameter { .. }
    )));
}
