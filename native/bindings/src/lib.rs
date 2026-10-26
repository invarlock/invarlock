//! Explicit opt-in Python companion; no automatic backend selection.
use invarlock_native_core as core;
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

fn error(err: core::IntervalError) -> PyErr {
    match err {
        core::IntervalError::IntermediateOverflow => PyOverflowError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

#[pyfunction]
#[pyo3(signature = (*, metric, baseline_scores, subject_scores, schedule_sha256))]
fn paired_resampling_interval(
    py: Python<'_>,
    metric: String,
    baseline_scores: Vec<f64>,
    subject_scores: Vec<f64>,
    schedule_sha256: String,
) -> PyResult<(f64, f64)> {
    py.detach(move || {
        core::paired_resampling_interval(
            &metric,
            &baseline_scores,
            &subject_scores,
            &schedule_sha256,
        )
    })
    .map_err(error)
}

#[pyfunction]
#[pyo3(signature = (*, metric, baseline_scores, subject_scores, schedule_sha256))]
fn resampling_distribution(
    py: Python<'_>,
    metric: String,
    baseline_scores: Vec<f64>,
    subject_scores: Vec<f64>,
    schedule_sha256: String,
) -> PyResult<Vec<f64>> {
    py.detach(move || {
        core::resampling_distribution(&metric, &baseline_scores, &subject_scores, &schedule_sha256)
    })
    .map_err(error)
}

#[pyfunction]
fn sample_indexes(
    py: Python<'_>,
    schedule_sha256: String,
    count: usize,
    replicate: u64,
) -> PyResult<Vec<usize>> {
    py.detach(move || core::sample_indexes(&schedule_sha256, count, replicate))
        .map_err(error)
}

#[pyfunction]
fn _diagnostic_fsum(py: Python<'_>, values: Vec<f64>) -> PyResult<f64> {
    py.detach(move || core::fsum(values)).map_err(error)
}

#[pymodule]
fn invarlock_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paired_resampling_interval, m)?)?;
    m.add_function(wrap_pyfunction!(resampling_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(sample_indexes, m)?)?;
    m.add_function(wrap_pyfunction!(_diagnostic_fsum, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
