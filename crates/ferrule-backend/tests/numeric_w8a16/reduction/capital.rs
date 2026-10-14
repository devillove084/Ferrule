//! Frozen external failure: M9=81, M32=288 violations before segmented WMMA.
const CAPITAL_PIN: &str = "b351ece2ac055fb4cf63bd1696190a90de0bbfb3f45c76724c908103e8418325";

#[test]
#[ignore = "GPU owner + original capital fixture; immutable manifest pin, fresh report required"]
fn capital_m9_m32_fixed_operator_bound() {
    let directory = std::env::var_os("FERRULE_W8A16_CAPITAL_CASE_DIR")
        .expect("path to the original capital-l1-qkv fixture");
    let report = std::env::var_os("FERRULE_W8A16_CAPITAL_REPORT").expect("fresh report path");
    // run verifies the fixed manifest and every payload BEFORE creating CUDA.
    // Keep the baseline gate unchanged, including the M8/M9 transition probes.
    super::captured::run(directory.into(), CAPITAL_PIN, report);
}
