//! Coarse Python boundaries. Immutable byte inputs stay borrowed while detached.
use invarlock_native_contracts as contracts;
use invarlock_native_core as analysis;
use invarlock_native_evidence as evidence;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use std::borrow::Cow;

// Only immutable Python bytes may be borrowed while the GIL is released. The
// caller retains their owner until detach returns. Mutable buffers and sequences
// keep the prior owned-snapshot extraction semantics.
fn request_bytes<'a>(raw: &'a Bound<'_, PyAny>) -> PyResult<Cow<'a, [u8]>> {
    if let Ok(bytes) = raw.cast::<PyBytes>() {
        Ok(Cow::Borrowed(bytes.as_bytes()))
    } else {
        raw.extract::<Vec<u8>>().map(Cow::Owned)
    }
}

pyo3::create_exception!(invarlock_native, NativeVerificationError, PyValueError);

fn evidence_error(py: Python<'_>, error: evidence::EvidenceError) -> PyErr {
    let err = NativeVerificationError::new_err(error.message);
    let value = err.value(py);
    let _ = value.setattr("exit_code", error.exit_code);
    let _ = value.setattr("receipt_written", error.receipt_written);
    err
}
#[pyfunction]
#[pyo3(signature=(raw, *, newline=true))]
fn canonicalize_json(
    py: Python<'_>,
    raw: &Bound<'_, PyAny>,
    newline: bool,
) -> PyResult<Py<PyBytes>> {
    let raw = request_bytes(raw)?;
    let bytes = py
        .detach(move || {
            let value = contracts::parse_json(&raw, "JSON input")?;
            if newline {
                contracts::canonical_json(&value)
            } else {
                contracts::canonical_json_no_newline(&value)
            }
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &bytes).unbind())
}
#[pyfunction]
fn validate_json_schema(py: Python<'_>, raw: &Bound<'_, PyAny>, schema: String) -> PyResult<()> {
    let raw = request_bytes(raw)?;
    py.detach(move || {
        let value = contracts::parse_json(&raw, "JSON input")?;
        contracts::validate_schema(&value, &schema)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn compare_runs_json(py: Python<'_>, raw: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let raw = request_bytes(raw)?;
    let result = py.detach(move || {
        let input = contracts::parse_json(&raw, "comparison request")
            .map_err(|e| (e.to_string(), "invalid_evaluation_records"))?;
        let maximum = contracts::bootstrap_budget(
            input.get("max_bootstrap_draws"),
            analysis::DEFAULT_MAX_BOOTSTRAP_DRAWS,
        )
        .map_err(|e| (e.to_string(), "invalid_evaluation_records"))?;
        let result = analysis::compare_runs(
            &input["baseline"],
            &input["subject"],
            &input["policy"],
            maximum,
        )
        .map_err(|e| (e.message, e.reason))?;
        contracts::canonical_json(&result)
            .map_err(|e| (e.to_string(), "invalid_evaluation_records"))
    });
    match result {
        Ok(bytes) => Ok(PyBytes::new(py, &bytes).unbind()),
        Err((message, reason)) => {
            let err = PyValueError::new_err(message);
            let _ = err.value(py).setattr("reason", reason);
            Err(err)
        }
    }
}
#[pyfunction]
fn verify_captured_json(py: Python<'_>, raw: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let raw = request_bytes(raw)?;
    let result = py.detach(move || {
        let config = contracts::parse_json(&raw, "verification request")?;
        let result = evidence::captured::verify(&config)?;
        contracts::canonical_json(&result).map_err(Into::into)
    });
    match result {
        Ok(bytes) => Ok(PyBytes::new(py, &bytes).unbind()),
        Err(error) => Err(evidence_error(py, error)),
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "NativeVerificationError",
        m.py().get_type::<NativeVerificationError>(),
    )?;
    m.add_function(wrap_pyfunction!(dispatch_json, m)?)?;
    m.add_function(wrap_pyfunction!(load_trust_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_json, m)?)?;
    m.add_function(wrap_pyfunction!(validate_json_schema, m)?)?;
    m.add_function(wrap_pyfunction!(compare_runs_json, m)?)?;
    m.add_function(wrap_pyfunction!(verify_captured_json, m)?)?;
    Ok(())
}

struct PythonScorer {
    callback: Py<PyAny>,
}
impl evidence::native_pack::ScorerReplay for PythonScorer {
    fn replay(&self, request: &serde_json::Value) -> evidence::Result<serde_json::Value> {
        let raw = contracts::canonical_json(request)?;
        let result = Python::attach(|py| -> Result<Vec<u8>, String> {
            self.callback
                .call1(py, (PyBytes::new(py, &raw),))
                .and_then(|result| result.extract::<Vec<u8>>(py))
                .map_err(|error| {
                    error
                        .value(py)
                        .str()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| "scorer callback failed".into())
                })
        })
        .map_err(evidence::EvidenceError::integrity)?;
        contracts::parse_json(&result, "authorized scorer result").map_err(Into::into)
    }
}

#[pyfunction]
#[pyo3(signature=(operation,raw,*,scorer_callback=None,policy_bytes=None,signing_key_bytes=None))]
fn dispatch_json(
    py: Python<'_>,
    operation: String,
    raw: &Bound<'_, PyAny>,
    scorer_callback: Option<Py<PyAny>>,
    policy_bytes: Option<Vec<u8>>,
    signing_key_bytes: Option<Vec<u8>>,
) -> PyResult<Py<PyBytes>> {
    let raw = request_bytes(raw)?;
    let outcome=py.detach(move ||->evidence::Result<Vec<u8>> {
        let config=contracts::parse_json(&raw,"native transaction request")?;
        let bridge=scorer_callback.map(|callback|PythonScorer{callback});
        let result=evidence::transaction::dispatch(&operation,&config,bridge.as_ref().map(|b|b as &dyn evidence::native_pack::ScorerReplay),policy_bytes.as_deref(),signing_key_bytes.as_deref())?;
        contracts::canonical_json(&serde_json::json!({"status":result.status,"payload":result.payload,"verified_report":result.verified_report})).map_err(Into::into)
    });
    match outcome {
        Ok(raw) => Ok(PyBytes::new(py, &raw).unbind()),
        Err(error) => Err(evidence_error(py, error)),
    }
}

type TrustSnapshot = (Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, bool);

#[pyfunction]
#[pyo3(signature=(path,key_override=None))]
fn load_trust_snapshot(
    py: Python<'_>,
    path: String,
    key_override: Option<Vec<u8>>,
) -> PyResult<TrustSnapshot> {
    let result = py
        .detach(move || {
            let trust =
                evidence::trust_inputs::load(std::path::Path::new(&path), key_override.as_deref())?;
            let raw = contracts::canonical_json(&trust.config)?;
            Ok::<_, evidence::EvidenceError>((
                raw,
                trust.policy_bytes,
                trust.signing_key_bytes,
                trust.captured,
            ))
        })
        .map_err(|error| evidence_error(py, error))?;
    Ok((
        PyBytes::new(py, &result.0).unbind(),
        PyBytes::new(py, &result.1).unbind(),
        PyBytes::new(py, &result.2).unbind(),
        result.3,
    ))
}
