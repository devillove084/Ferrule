use super::*;
use serde::Deserialize;
use serde::de::{
    IntoDeserializer, Visitor,
    value::{Error, MapDeserializer, SeqDeserializer},
};

// Exercise serde's actual DTO contract without adding a JSON dependency to common.
#[derive(Clone)]
enum Value {
    Null,
    Number(u64),
    Text(String),
    List(Vec<Value>),
    Map(Vec<(String, Value)>),
}
impl<'de> IntoDeserializer<'de, Error> for Value {
    type Deserializer = Self;
    fn into_deserializer(self) -> Self {
        self
    }
}
impl<'de> serde::Deserializer<'de> for Value {
    type Error = Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Self::Null => visitor.visit_unit(),
            Self::Number(v) => visitor.visit_u64(v),
            Self::Text(v) => visitor.visit_string(v),
            Self::List(v) => visitor.visit_seq(SeqDeserializer::new(v.into_iter())),
            Self::Map(v) => visitor.visit_map(MapDeserializer::new(v.into_iter())),
        }
    }
    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        if matches!(self, Self::Null) {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        match self {
            Self::Text(v) => visitor.visit_enum(v.into_deserializer()),
            other => other.deserialize_any(visitor),
        }
    }
    serde::forward_to_deserialize_any! { bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf unit unit_struct newtype_struct seq tuple tuple_struct map struct identifier ignored_any }
}
fn text(v: &str) -> Value {
    Value::Text(v.into())
}
fn map(v: Vec<(&str, Value)>) -> Value {
    Value::Map(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn wire_demand() -> Value {
    map(vec![
        ("type", text("demand")),
        ("seq", Value::Number(0)),
        (
            "request",
            map(vec![("id", text("r")), ("generation", Value::Number(0))]),
        ),
        ("namespace_id", text("n")),
        (
            "prompt_tokens",
            Value::List(vec![Value::Number(1), Value::Number(2)]),
        ),
        ("reuse_limit", text("all_prompt_tokens")),
    ])
}

#[test]
fn flat_tagged_demand_dto_decodes() {
    assert_eq!(
        TraceEvent::deserialize(wire_demand()).unwrap(),
        demand(0, "r", &[1, 2])
    );
}
#[test]
fn strict_unknown_duplicate_missing_and_invalid_fields() {
    for key in ["seq", "unrecognized"] {
        let Value::Map(mut fields) = wire_demand() else {
            unreachable!()
        };
        fields.push((key.into(), Value::Number(0)));
        assert!(
            TraceEvent::deserialize(Value::Map(fields)).is_err(),
            "{key}"
        );
    }
    let Value::Map(mut fields) = wire_demand() else {
        unreachable!()
    };
    fields.retain(|(key, _)| key != "namespace_id");
    assert!(TraceEvent::deserialize(Value::Map(fields)).is_err());
    let Value::Map(mut fields) = wire_demand() else {
        unreachable!()
    };
    fields
        .iter_mut()
        .find(|(key, _)| key == "prompt_tokens")
        .unwrap()
        .1 = Value::List(vec![Value::Number(u64::MAX)]);
    assert!(TraceEvent::deserialize(Value::Map(fields)).is_err());
    assert!(AnalyzerConfig::deserialize(map(vec![("unknown", Value::Number(0))])).is_err());
    assert!(
        Ratio::deserialize(map(vec![
            ("numerator", Value::Number(1)),
            ("denominator", Value::Number(2)),
            ("extra", Value::Number(0))
        ]))
        .is_err()
    );
}
#[test]
fn config_defaults_and_null_actual() {
    assert_eq!(
        AnalyzerConfig::deserialize(map(vec![])).unwrap(),
        AnalyzerConfig::default()
    );
    let value = map(vec![
        ("type", text("outcome")),
        ("seq", Value::Number(2)),
        (
            "request",
            map(vec![("id", text("r")), ("generation", Value::Number(0))]),
        ),
        ("status", text("cancelled")),
        ("actual", Value::Null),
    ]);
    assert_eq!(
        TraceEvent::deserialize(value).unwrap(),
        outcome(2, "r", RequestStatus::Cancelled)
    );
}
