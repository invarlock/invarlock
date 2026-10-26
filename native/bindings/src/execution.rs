//! Explicit execution bindings; transport work releases the interpreter lock.
use invarlock_native_execution as execution;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
#[pyclass(skip_from_py_object)]
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}
#[pymethods]
impl CancellationToken {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}
#[pyfunction]
#[allow(clippy::too_many_arguments)] // Explicit keyword-only process capabilities are part of the Python API.
#[pyo3(signature=(program,args,input,timeout_seconds,stdout_limit,stderr_limit,*,cancellation=None,cwd=None,environment=None,pass_fds=None))]
fn run_process<'py>(
    py: Python<'py>,
    program: String,
    args: Vec<String>,
    input: Vec<u8>,
    timeout_seconds: f64,
    stdout_limit: usize,
    stderr_limit: usize,
    cancellation: Option<PyRef<'_, CancellationToken>>,
    cwd: Option<String>,
    environment: Option<std::collections::BTreeMap<String, String>>,
    pass_fds: Option<Vec<i32>>,
) -> PyResult<(i32, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    if !timeout_seconds.is_finite() || timeout_seconds <= 0.0 || timeout_seconds > 86400.0 {
        return Err(PyValueError::new_err("invalid process timeout"));
    }
    let token = cancellation
        .map(|v| v.cancelled.clone())
        .unwrap_or_default();
    let spec = execution::ProcessSpec {
        program: program.into(),
        args: args.into_iter().map(Into::into).collect(),
        input,
        timeout: Duration::from_secs_f64(timeout_seconds),
        stdout_limit,
        stderr_limit,
        env: environment.map(|values| {
            values
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect()
        }),
        cwd: cwd.map(Into::into),
    };
    let descriptors = pass_fds.unwrap_or_default();
    let result = py
        .detach(move || execution::run_process_with_descriptors(&spec, &token, &descriptors))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        result.status,
        PyBytes::new(py, &result.stdout),
        PyBytes::new(py, &result.stderr),
    ))
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CancellationToken>()?;
    module.add_class::<InspectCheckpoint>()?;
    module.add_class::<CollectionCoordinator>()?;
    module.add_class::<SequentialCoordinator>()?;
    module.add_class::<Workspace>()?;
    module.add_function(wrap_pyfunction!(run_process, module)?)?;
    module.add_function(wrap_pyfunction!(run_oci_workers, module)?)?;
    module.add_function(wrap_pyfunction!(run_oci_command, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_oci_inspection, module)?)?;
    Ok(())
}

#[pyclass]
pub struct InspectCheckpoint {
    inner: std::sync::Mutex<Option<execution::inspect_checkpoint::InspectCheckpoint>>,
}
impl InspectCheckpoint {
    fn with<T>(
        &self,
        f: impl FnOnce(&mut execution::inspect_checkpoint::InspectCheckpoint) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| "checkpoint lock was poisoned".to_string())?;
        f(guard
            .as_mut()
            .ok_or_else(|| "checkpoint is closed".to_string())?)
    }
}
#[pymethods]
impl InspectCheckpoint {
    #[new]
    fn new(
        py: Python<'_>,
        directory: String,
        header_json: Vec<u8>,
        empty_export_json: Vec<u8>,
    ) -> PyResult<Self> {
        let checkpoint = py
            .detach(move || -> Result<_, String> {
                let header = invarlock_native_contracts::parse_json(
                    &header_json,
                    "Inspect checkpoint header",
                )
                .map_err(|e| e.to_string())?;
                let empty = invarlock_native_contracts::parse_json(
                    &empty_export_json,
                    "Inspect empty export",
                )
                .map_err(|e| e.to_string())?;
                execution::inspect_checkpoint::InspectCheckpoint::open(
                    std::path::Path::new(&directory),
                    &header,
                    &empty,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(Some(checkpoint)),
        })
    }
    fn export_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let raw = py
            .detach(|| {
                self.with(|checkpoint| {
                    let exported = checkpoint.export().map_err(|e| e.to_string())?;
                    invarlock_native_contracts::canonical_json_no_newline(&exported)
                        .map_err(|e| e.to_string())
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &raw))
    }
    fn admit(
        &self,
        py: Python<'_>,
        trial_id: String,
        attempt: u64,
        event_json: Vec<u8>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.with(|checkpoint| {
                let event =
                    invarlock_native_contracts::parse_json(&event_json, "Inspect admission event")
                        .map_err(|e| e.to_string())?;
                checkpoint
                    .admit(&trial_id, attempt, &event)
                    .map_err(|e| e.to_string())
            })
        })
        .map_err(PyValueError::new_err)
    }
    fn complete(
        &self,
        py: Python<'_>,
        trial_id: String,
        attempt: u64,
        event_json: Vec<u8>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.with(|checkpoint| {
                let event =
                    invarlock_native_contracts::parse_json(&event_json, "Inspect completed event")
                        .map_err(|e| e.to_string())?;
                checkpoint
                    .complete(&trial_id, attempt, &event)
                    .map_err(|e| e.to_string())
            })
        })
        .map_err(PyValueError::new_err)
    }
    fn pending_admissions(&self, py: Python<'_>) -> PyResult<Vec<(String, u64)>> {
        py.detach(|| self.with(|checkpoint| Ok(checkpoint.pending_admissions())))
            .map_err(PyValueError::new_err)
    }
    fn close(&self) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature=(image,digest,inspection_json,*,allow_tag=true))]
fn resolve_oci_inspection(
    py: Python<'_>,
    image: String,
    digest: String,
    inspection_json: Vec<u8>,
    allow_tag: bool,
) -> PyResult<String> {
    py.detach(move || {
        let inspection = execution::oci::ImageInspection::parse(&inspection_json)?;
        inspection.resolve(&image, &digest, allow_tag)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass]
pub struct CollectionCoordinator {
    inner: std::sync::Mutex<Option<execution::collection::CollectionCoordinator>>,
    cancel: Arc<AtomicBool>,
}
impl CollectionCoordinator {
    fn with<T>(
        &self,
        f: impl FnOnce(&mut execution::collection::CollectionCoordinator) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| "collection state lock was poisoned".to_string())?;
        f(guard
            .as_mut()
            .ok_or_else(|| "collection is closed".to_string())?)
    }
    fn json<'py>(
        &self,
        py: Python<'py>,
        f: impl FnOnce(
                &mut execution::collection::CollectionCoordinator,
            ) -> Result<invarlock_native_contracts::Value, String>
            + Send,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let raw = py
            .detach(|| {
                self.with(|state| {
                    let value = f(state)?;
                    invarlock_native_contracts::canonical_json_no_newline(&value)
                        .map_err(|e| e.to_string())
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &raw))
    }
}
#[pymethods]
impl CollectionCoordinator {
    #[new]
    #[pyo3(signature=(directory,plan_json,options_json,baseline_json,subject_json,scorer_id,invocation_timeout_seconds,*,stop_after_batches=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        directory: String,
        plan_json: Vec<u8>,
        options_json: Vec<u8>,
        baseline_json: Vec<u8>,
        subject_json: Vec<u8>,
        scorer_id: String,
        invocation_timeout_seconds: u64,
        stop_after_batches: Option<u64>,
    ) -> PyResult<Self> {
        let state = py
            .detach(move || -> Result<_, String> {
                let parse = |raw: &[u8], label: &str| {
                    invarlock_native_contracts::parse_json(raw, label).map_err(|e| e.to_string())
                };
                execution::collection::CollectionCoordinator::open(
                    std::path::Path::new(&directory),
                    parse(&plan_json, "judge plan")?,
                    parse(&options_json, "collection options")?,
                    parse(&baseline_json, "baseline frozen run")?,
                    parse(&subject_json, "subject frozen run")?,
                    &scorer_id,
                    Duration::from_secs(invocation_timeout_seconds),
                    stop_after_batches,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        let cancel = state.cancellation();
        Ok(Self {
            inner: std::sync::Mutex::new(Some(state)),
            cancel,
        })
    }
    fn begin_batch_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.json(py, |state| {
            Ok(invarlock_native_contracts::Value::Array(
                state
                    .begin_batch()
                    .map_err(|e| e.to_string())?
                    .iter()
                    .map(|call| call.to_value())
                    .collect(),
            ))
        })
    }
    fn wait_start(&self, py: Python<'_>, trial_id: String) -> PyResult<()> {
        py.detach(|| {
            let pacer =
                self.with(|state| state.prepare_start(&trial_id).map_err(|e| e.to_string()))?;
            pacer.wait(&self.cancel).map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.with(|state| state.verify().map_err(|e| e.to_string())))
            .map_err(PyValueError::new_err)
    }
    fn complete(&self, py: Python<'_>, trial_id: String, event_json: Vec<u8>) -> PyResult<()> {
        py.detach(|| {
            self.with(|state| {
                let event =
                    invarlock_native_contracts::parse_json(&event_json, "completed SDK event")
                        .map_err(|e| e.to_string())?;
                state.complete(&trial_id, &event).map_err(|e| e.to_string())
            })
        })
        .map_err(PyValueError::new_err)
    }
    fn remaining_seconds(&self) -> PyResult<f64> {
        self.with(|state| Ok(state.remaining_seconds()))
            .map_err(PyValueError::new_err)
    }
    fn sdk_config_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.json(py, |state| {
            state.sdk_configuration().map_err(|e| e.to_string())
        })
    }
    fn measurements_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.json(py, |state| state.measurements().map_err(|e| e.to_string()))
    }
    fn status_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.json(py, |state| state.status().map_err(|e| e.to_string()))
    }
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    fn stop(&self, py: Python<'_>, reason: String) -> PyResult<()> {
        py.detach(|| self.with(|state| state.stop(&reason).map_err(|e| e.to_string())))
            .map_err(PyValueError::new_err)
    }
    fn close(&self) -> PyResult<()> {
        self.cancel.store(true, Ordering::Release);
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature=(request_json,*,cancellation=None))]
fn run_oci_workers<'py>(
    py: Python<'py>,
    request_json: Vec<u8>,
    cancellation: Option<PyRef<'_, CancellationToken>>,
) -> PyResult<Bound<'py, PyBytes>> {
    if request_json.len() > 16 * 1024 * 1024 {
        return Err(PyValueError::new_err("worker request exceeds 16 MiB"));
    }
    let token = cancellation
        .map(|v| v.cancelled.clone())
        .unwrap_or_default();
    let result = py
        .detach(move || {
            let request =
                invarlock_native_contracts::parse_json(&request_json, "OCI worker request")
                    .map_err(|e| e.to_string())?;
            let result =
                execution::oci::run_worker_pair(&request, &token).map_err(|e| e.to_string())?;
            serde_json::to_vec(&result).map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &result))
}

#[pyclass]
pub struct SequentialCoordinator {
    inner: std::sync::Mutex<Option<execution::sequential::SequentialCoordinator>>,
}
impl SequentialCoordinator {
    fn with<T>(
        &self,
        f: impl FnOnce(&mut execution::sequential::SequentialCoordinator) -> execution::Result<T>,
    ) -> PyResult<T> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("sequential coordinator lock was poisoned"))?;
        f(guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("sequential coordinator is closed"))?)
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
#[pymethods]
impl SequentialCoordinator {
    #[new]
    fn new(
        py: Python<'_>,
        directory: String,
        identity_json: Vec<u8>,
        plan_json: Vec<u8>,
        baseline_json: Vec<u8>,
        subject_json: Vec<u8>,
    ) -> PyResult<Self> {
        let inner = py
            .detach(move || {
                let parse = |raw: &[u8]| {
                    if raw.len() > 128 * 1024 * 1024 {
                        return Err("sequential input exceeds byte limit".to_string());
                    }
                    invarlock_native_contracts::parse_json(raw, "sequential input")
                        .map_err(|e| e.to_string())
                };
                execution::sequential::SequentialCoordinator::open(
                    std::path::Path::new(&directory),
                    parse(&identity_json)?,
                    parse(&plan_json)?,
                    parse(&baseline_json)?,
                    parse(&subject_json)?,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(Some(inner)),
        })
    }
    fn sources_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let value = py.detach(|| self.with(|s| s.sources()))?;
        let raw = serde_json::to_vec(&value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &raw))
    }
    fn admit<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        request: Vec<u8>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let value = py.detach(|| self.with(|s| s.admit(index, &request)))?;
        let raw = serde_json::to_vec(&value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &raw))
    }
    fn complete(&self, py: Python<'_>, index: usize, source_json: Vec<u8>) -> PyResult<()> {
        if source_json.len() > 16 * 1024 * 1024 {
            return Err(PyValueError::new_err("retained source exceeds byte limit"));
        }
        let source = invarlock_native_contracts::parse_json(&source_json, "sequential outcome")
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| self.with(|s| s.complete(index, source)))
    }
    fn finish_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let value = py.detach(|| self.with(|s| s.finish()))?;
        let raw = serde_json::to_vec(&value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &raw))
    }
    fn close(&self) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.take();
    }
}

#[pyclass]
pub struct Workspace {
    inner: std::sync::Mutex<Option<execution::workspace::Workspace>>,
}
fn workspace_error(error: execution::Error) -> PyErr {
    let kind = match &error {
        execution::Error::Io(e)
        | execution::Error::Filesystem(invarlock_native_filesystem::Error::Io(e)) => Some(e.kind()),
        _ => None,
    };
    match kind {
        Some(std::io::ErrorKind::NotFound) => {
            pyo3::exceptions::PyFileNotFoundError::new_err(error.to_string())
        }
        Some(std::io::ErrorKind::AlreadyExists) => {
            pyo3::exceptions::PyFileExistsError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}
impl Workspace {
    fn with<T>(
        &self,
        f: impl FnOnce(&mut execution::workspace::Workspace) -> execution::Result<T>,
    ) -> PyResult<T> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("workspace lock was poisoned"))?;
        f(guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("workspace is closed"))?)
        .map_err(workspace_error)
    }
}
#[pymethods]
impl Workspace {
    #[new]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        let inner = py
            .detach(|| execution::workspace::Workspace::open(std::path::Path::new(&path)))
            .map_err(workspace_error)?;
        Ok(Self {
            inner: std::sync::Mutex::new(Some(inner)),
        })
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.with(|s| s.verify()))
    }
    fn read<'py>(
        &self,
        py: Python<'py>,
        name: String,
        limit: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let raw = py.detach(|| self.with(|s| s.read(&name, limit)))?;
        Ok(PyBytes::new(py, &raw))
    }
    fn write(&self, py: Python<'_>, name: String, payload: Vec<u8>) -> PyResult<()> {
        py.detach(|| self.with(|s| s.write(&name, &payload)))
    }
    fn retain_identity(&self, py: Python<'_>, name: String, value: Vec<u8>) -> PyResult<()> {
        py.detach(|| self.with(|s| s.retain_identity(&name, &value)))
    }
    fn close(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}

#[pyfunction]
fn run_oci_command<'py>(
    py: Python<'py>,
    command: Vec<String>,
    timeout_seconds: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    if command.is_empty() || !(1..=86400).contains(&timeout_seconds) {
        return Err(PyValueError::new_err(
            "invalid OCI cleanup command or deadline",
        ));
    }
    let spec = execution::ProcessSpec {
        program: command[0].clone().into(),
        args: command.into_iter().skip(1).map(Into::into).collect(),
        input: vec![],
        timeout: Duration::from_secs(timeout_seconds),
        stdout_limit: 65536,
        stderr_limit: 65536,
        env: None,
        cwd: None,
    };
    let result = py
        .detach(|| execution::oci::run_side_worker(&spec, &AtomicBool::new(false)))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let raw=serde_json::to_vec(&serde_json::json!({"status":result.output.status,"stdout":String::from_utf8_lossy(&result.output.stdout),"stderr":String::from_utf8_lossy(&result.output.stderr),"cleanup_errors":result.cleanup_errors})).map_err(|e|PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &raw))
}
