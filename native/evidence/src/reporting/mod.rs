//! Non-authoritative presentations of explicitly supplied comparison facts.
//! Rendering never authorizes a signer or substitutes for an evidence replay.
mod markdown;
mod presentation;
pub use markdown::render_markdown;
pub use presentation::{display_label, number, terminal_text, visible_controls};

mod html;
pub use html::render_html;

mod junit;
pub use junit::render_junit;

pub(crate) mod captured_view;
pub use captured_view::build_comparison_view;
mod native_view;
pub use native_view::build_native_view;

mod context;

pub use captured_view::load as load_captured_report;

pub(crate) mod judge_view;
pub mod publication;
pub mod sets;
pub use judge_view::load as load_judge_report;

mod native_checks;
mod native_load;
pub use native_load::load as load_native_report;

mod transaction;
pub use transaction::render;

pub mod handoff;
