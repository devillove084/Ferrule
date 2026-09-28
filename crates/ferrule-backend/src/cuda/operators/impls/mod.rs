//! Semantic family execution, intentionally outside context-child privacy.
//! Context resources are accessed only through scoped submission and borrows.
mod attention;
mod compressor;
mod hc;
mod moe;
mod projection;
mod proposal;
