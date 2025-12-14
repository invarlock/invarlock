"""
InvarLock CLI Doctor Command
========================

Handles the 'invarlock doctor' command for health checks.
"""

import importlib.util
import os as _os
import platform as _platform
import shutil as _shutil
import sys
import warnings
from collections.abc import Callable

import typer
from rich.console import Console
from rich.table import Table

from ..constants import DOCTOR_FORMAT_VERSION

# Exact wording constant for determinism warning (kept in one place)
DETERMINISM_SHARDS_WARNING = "Provider workers > 0 without deterministic_shards=True; enable deterministic_shards or set workers=0 for determinism."

console = Console()


def _cross_check_reports(
    baseline_report: str | None,
    subject_report: str | None,
    *,
    cfg_metric_kind: str | None,
    strict: bool,
    profile: str | None,
    json_out: bool,
    console: Console,
    add_fn: Callable[..., None],
) -> bool:
    """Perform baseline vs subject cross-checks and report findings."""
    had_error = False
    try:
        import json as _json_cc
        from pathlib import Path as _Path_cc

        if baseline_report and subject_report:
            bpath = _Path_cc(baseline_report)
            spath = _Path_cc(subject_report)
            if bpath.exists() and spath.exists():
                bdata = _json_cc.loads(bpath.read_text())
                sdata = _json_cc.loads(spath.read_text())
                bprov = bdata.get("provenance", {}) if isinstance(bdata, dict) else {}
                sprov = sdata.get("provenance", {}) if isinstance(sdata, dict) else {}
                # D009: tokenizer digest mismatch
                try:
                    bdig = bprov.get("provider_digest", {}) or {}
                    sdig = sprov.get("provider_digest", {}) or {}
                    btok = bdig.get("tokenizer_sha256")
                    stok = sdig.get("tokenizer_sha256")
                    if (
                        isinstance(btok, str)
                        and isinstance(stok, str)
                        and btok
                        and stok
                        and btok != stok
                    ):
                        add_fn(
                            "D009",
                            "warning",
                            "tokenizer digests differ between baseline and subject; run will abort in ci/release (E002).",
                            field="provenance.provider_digest.tokenizer_sha256",
                        )
                except Exception:
                    pass
                # D010: MLM mask digest missing (only for ppl_mlm)
                try:
                    bdig = bprov.get("provider_digest", {}) or {}
                    sdig = sprov.get("provider_digest", {}) or {}
                    btok = bdig.get("tokenizer_sha256")
                    stok = sdig.get("tokenizer_sha256")
                    bmask = bdig.get("masking_sha256")
                    smask = sdig.get("masking_sha256")
                    # Determine if PM is MLM from either report or config context
                    try:
                        pm_b = (
                            (bdata.get("metrics", {}) or {}).get("primary_metric", {})
                            or {}
                        ).get("kind")
                        pm_s = (
                            (sdata.get("metrics", {}) or {}).get("primary_metric", {})
                            or {}
                        ).get("kind")
                    except Exception:
                        pm_b = pm_s = None
                    is_mlm = False
                    for _k in (pm_b, pm_s, cfg_metric_kind):
                        try:
                            if isinstance(_k, str) and _k.lower() == "ppl_mlm":
                                is_mlm = True
                                break
                        except Exception:
                            pass
                    if (
                        is_mlm
                        and isinstance(btok, str)
                        and isinstance(stok, str)
                        and btok
                        and stok
                        and btok == stok
                        and (not bmask or not smask)
                    ):
                        add_fn(
                            "D010",
                            "warning",
                            "ppl_mlm with matching tokenizer but missing masking digests; ci/release may abort on mask parity.",
                            baseline_has_mask=bool(bmask),
                            subject_has_mask=bool(smask),
                        )
                except Exception:
                    pass
                # D011: split mismatch
                try:
                    bsplit = bprov.get("dataset_split")
                    ssplit = sprov.get("dataset_split")
                    if (
                        isinstance(bsplit, str)
                        and isinstance(ssplit, str)
                        and bsplit
                        and ssplit
                        and bsplit != ssplit
                    ):
                        sev = "error" if bool(strict) else "warning"
                        add_fn(
                            "D011",
                            sev,
                            f"dataset split mismatch (baseline={bsplit}, subject={ssplit})",
                            field="provenance.dataset_split",
                            baseline=bsplit,
                            subject=ssplit,
                        )
                        if sev == "error":
                            had_error = True
                except Exception:
                    pass
                # D012: Accuracy PM flagged as estimated/pseudo (warn in dev; error in ci/release)
                try:
                    spm = (sdata.get("metrics", {}) or {}).get(
                        "primary_metric", {}
                    ) or {}
                    kind = str(spm.get("kind", "")).lower()
                    if kind in {"accuracy", "vqa_accuracy"}:
                        estimated = bool(spm.get("estimated"))
                        counts_source = str(spm.get("counts_source", "")).lower()
                        if estimated or counts_source == "pseudo_config":
                            prof = None
                            try:
                                prof = str(
                                    (sdata.get("meta", {}) or {}).get("profile", "")
                                ).lower()
                            except Exception:
                                prof = None
                            prof_flag = None
                            try:
                                prof_flag = str(profile).lower() if profile else None
                            except Exception:
                                prof_flag = None
                            eff_prof = prof_flag or prof or "dev"
                            sev = "warning" if eff_prof == "dev" else "error"
                            add_fn(
                                "D012",
                                sev,
                                "accuracy primary metric uses pseudo/estimated counts; use labeled preset for measured accuracy.",
                                field="metrics.primary_metric",
                            )
                            if sev == "error":
                                had_error = True
                except Exception:
                    pass
    except Exception:
        pass
    return had_error


DATASET_SPLIT_FALLBACK_WARNING = "Dataset split was inferred via fallback; set dataset.split explicitly to avoid drift."


def doctor_command(
    config: str | None = typer.Option(
        None, "--config", "-c", help="Path to YAML config for preflight lints"
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Profile to apply for preflight (ci|release)"
    ),
    baseline: str | None = typer.Option(
        None, "--baseline", help="Optional baseline report to check pairing readiness"
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON (suppresses human-readable output)",
    ),
    tier: str | None = typer.Option(
        None,
        "--tier",
        help="Policy tier for floors preview (conservative|balanced|aggressive)",
    ),
    baseline_report: str | None = typer.Option(
        None, "--baseline-report", help="Optional baseline report for cross-checks"
    ),
    subject_report: str | None = typer.Option(
        None, "--subject-report", help="Optional subject report for cross-checks"
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Escalate certain warnings (e.g., split mismatch) to errors",
    ),
):
    """
    Perform health checks on InvarLock installation.

    Checks PyTorch, device availability, memory, and optional extras.
    """

    # Normalize Typer OptionInfo placeholders when invoked directly in tests
    def _is_optioninfo_like(obj: object) -> bool:
        try:
            # True for Typer's OptionInfo; robust to import shims/mocks
            cname = getattr(obj, "__class__", type(None)).__name__
            if cname == "OptionInfo":
                return True
            # Heuristic: has typical Typer OptionInfo attributes
            return hasattr(obj, "param_decls") and hasattr(obj, "default")
        except Exception:
            return False

    def _coerce_opt(val: object, *, bool_default: bool | None = None):
        if _is_optioninfo_like(val):
            if isinstance(bool_default, bool):
                return bool_default
            return None
        return val

    config = _coerce_opt(config)
    profile = _coerce_opt(profile)
    baseline = _coerce_opt(baseline)
    tier = _coerce_opt(tier)
    baseline_report = _coerce_opt(baseline_report)
    subject_report = _coerce_opt(subject_report)
    strict = bool(_coerce_opt(strict, bool_default=False))
    json_out = bool(_coerce_opt(json_out, bool_default=False))

    # Findings accumulator for --json mode
    findings: list[dict] = []

    def _add(code: str, severity: str, message: str, **extra: object) -> None:
        item = {"code": code, "severity": severity, "message": message}
        if extra:
            item.update(extra)
        findings.append(item)
        if not json_out:
            prefix = (
                "ERROR:"
                if severity == "error"
                else ("WARNING:" if severity == "warning" else "NOTE:")
            )
            typer.echo(f"{prefix} {message} [INVARLOCK:{code}]")

    # Early: surface tiny relax as a note when active (env-based)
    try:
        import os as __os

        if str(__os.environ.get("INVARLOCK_TINY_RELAX", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _add(
                "D013",
                "note",
                "tiny relax (dev) active; gates widened and drift/overhead may be informational.",
                field="auto.tiny_relax",
            )
    except Exception:
        pass

    # Redirect rich Console output in JSON mode so no extra text is emitted
    if json_out:
        from io import StringIO

        global console
        console = Console(file=StringIO())

    if not json_out:
        console.print("🏥 InvarLock Health Check")
        console.print("=" * 50)

    # Environment facts (OS · Python · invarlock)
    try:
        from invarlock import __version__ as _invarlock_version  # type: ignore
    except Exception:
        _invarlock_version = "unknown"
    if not json_out:
        os_line = (
            f"OS: {_platform.system()} {_platform.release()} ({_platform.machine()})"
        )
        py_line = f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        console.print(f"{os_line} · {py_line} · invarlock: {_invarlock_version}")

    health_status = True
    had_error = False
    cfg_metric_kind: str | None = None

    # Check core components
    try:
        from invarlock.core.registry import get_registry

        if not json_out:
            console.print("[green]✅ Core components available[/green]")
    except ImportError as e:
        if not json_out:
            console.print(f"[red]❌ Core components missing: {e}[/red]")
        health_status = False

    # Check PyTorch
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        if not json_out:
            if torch_version:
                console.print(f"[green]✅ PyTorch {torch_version}[/green]")
            else:
                console.print(
                    "[yellow]⚠️  PyTorch present but version unavailable[/yellow]"
                )

        # Device information
        from ..device import get_device_info

        device_info = get_device_info()
        if not json_out:
            console.print("\n🖥️  Device Information")

        for device_name, info in device_info.items():
            if device_name == "auto_selected":
                if not json_out:
                    console.print(f"  ▶ Auto‑selected device: {info}")
                continue

            if info["available"]:
                if (
                    device_name == "cuda"
                    and isinstance(info, dict)
                    and "device_count" in info
                ):
                    if not json_out:
                        console.print(
                            f"  [green]✅ {device_name.upper()}: {info['device_count']} device(s) - {info['device_name']} ({info['memory_total']})[/green]"
                        )
                else:
                    if not json_out:
                        console.print(
                            f"  [green]✅ {device_name.upper()}: Available[/green]"
                        )
            else:
                if not json_out:
                    console.print(
                        f"  [dim]❌ {device_name.upper()}: {info['info']}[/dim]"
                    )

        # CUDA triage details
        try:
            cuda_toolkit_found = bool(
                _shutil.which("nvcc") or _shutil.which("nvidia-smi")
            )
            torch_cuda_build = bool(getattr(torch.version, "cuda", None))
            cuda_available = bool(
                getattr(torch, "cuda", None) and torch.cuda.is_available()
            )
            if not json_out:
                console.print(
                    f"  [dim]• CUDA toolkit: {'found' if cuda_toolkit_found else 'not found'} · "
                    f"torch CUDA build: {'yes' if torch_cuda_build else 'no'} · "
                    f"cuda.is_available(): {'true' if cuda_available else 'false'}[/dim]"
                )
        except Exception:
            pass

        # Memory check
        try:
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if not json_out:
                    console.print(f"\n💾 GPU Memory: {free_memory:.1f} GB total")
                if free_memory < 4.0:
                    if not json_out:
                        console.print(
                            "[yellow]⚠️  Warning: Less than 4GB GPU memory available[/yellow]"
                        )
        except Exception:
            pass

    except ImportError:
        if not json_out:
            console.print("[red]❌ PyTorch not available[/red]")
            console.print("Install with: pip install torch")
        health_status = False

    # Check optional dependencies
    if not json_out:
        console.print("\n📦 Optional Dependencies")

    optional_deps = [
        ("datasets", "Dataset loading (WikiText-2, etc.)"),
        ("transformers", "Hugging Face model support"),
        ("auto_gptq", "GPTQ quantization (Linux/CUDA only)"),
        ("autoawq", "AWQ quantization (Linux/CUDA only)"),
        ("bitsandbytes", "8/4-bit loading (GPU)"),
    ]

    # Query CUDA availability once
    try:
        import torch as _torch

        has_cuda = bool(getattr(_torch, "cuda", None) and _torch.cuda.is_available())
    except Exception:
        has_cuda = False

    for dep, description in optional_deps:
        spec = importlib.util.find_spec(dep)
        present = spec is not None
        extra_hint = {
            "datasets": "eval",
            "transformers": "adapters",
            "auto_gptq": "gptq",
            "autoawq": "awq",
            "bitsandbytes": "gpu",
        }.get(dep, dep)

        if dep == "bitsandbytes":
            # Avoid importing bnb to suppress noisy CPU-only warnings. Report based on CUDA.
            if not has_cuda:
                # GPU-only library; note and skip import
                if present:
                    if not json_out:
                        console.print(
                            "  [yellow]⚠️  bitsandbytes — CUDA-only; GPU not detected on this host[/yellow]"
                        )
                else:
                    if not json_out:
                        console.print(
                            "  [dim]⚠️  bitsandbytes — CUDA-only; not installed[/dim]"
                        )
                        console.print(
                            "     → Install: pip install 'invarlock[gpu]'",
                            markup=False,
                        )
            else:
                # CUDA available; try a quiet import and detect CPU-only builds
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                    if not json_out:
                        console.print(
                            "  [green]✅ bitsandbytes — 8/4-bit loading (GPU)[/green]"
                        )
                except Exception:
                    if not json_out:
                        console.print(
                            "  [yellow]⚠️  bitsandbytes — Present but CPU-only build detected[/yellow]"
                        )
                        console.print(
                            "     → Reinstall with: pip install 'invarlock[gpu]' on a CUDA host",
                            markup=False,
                        )
            continue

        if not json_out:
            if present:
                console.print(f"  [green]✅ {dep} — {description}[/green]")
            else:
                console.print(f"  [yellow]⚠️  {dep} — {description}[/yellow]")
                # Remediation for platform-gated stacks
                if dep in {"auto_gptq", "autoawq"}:
                    console.print(
                        f"     → Install: pip install 'invarlock[{extra_hint}]'  # Linux + CUDA only",
                        markup=False,
                    )
                else:
                    console.print(
                        f"     → Install: pip install 'invarlock[{extra_hint}]'",
                        markup=False,
                    )

    # Optional: Config preflight (determinism & provider)
    if config:
        console.print("\n🧪 Preflight Lints (config)")
        try:
            import json as _json
            from pathlib import Path

            from invarlock.eval.data import get_provider
            from invarlock.model_profile import detect_model_profile, resolve_tokenizer

            from ..commands.run import _resolve_metric_and_provider
            from ..config import apply_profile, load_config

            cfg = load_config(config)
            if profile:
                try:
                    cfg = apply_profile(cfg, profile)
                    console.print(f"  ▶ Profile applied: {profile}")
                except Exception as _e:
                    console.print(f"  [yellow]⚠️ Profile apply failed: {_e}[/yellow]")

            # Provider kind sanity (D001)
            try:
                SUPPORTED_PROVIDERS = {
                    "wikitext2",
                    "hf_text",
                    "synthetic",
                    "local_jsonl",
                }
                provider_cfg = getattr(
                    getattr(cfg, "dataset", object()), "provider", None
                )
                bad_kind: str | None = None

                # Helper to read mapping-like config values
                def _pget(obj, key: str) -> str | None:
                    try:
                        if isinstance(obj, dict):
                            return obj.get(key)  # type: ignore[return-value]
                        # support mapping-like config objects (_Obj)
                        if hasattr(obj, key):
                            return getattr(obj, key)  # type: ignore[return-value]
                        get = getattr(obj, "get", None)
                        if callable(get):  # type: ignore[call-arg]
                            return get(key)  # type: ignore[return-value]
                    except Exception:
                        return None
                    return None

                if isinstance(provider_cfg, dict):
                    k = str(provider_cfg.get("kind", "")).strip()
                    if not k or k not in SUPPORTED_PROVIDERS:
                        bad_kind = k or ""
                elif isinstance(provider_cfg, str):
                    if provider_cfg not in SUPPORTED_PROVIDERS:
                        bad_kind = provider_cfg
                else:
                    k2 = str(_pget(provider_cfg, "kind") or "").strip()
                    if not k2 or k2 not in SUPPORTED_PROVIDERS:
                        bad_kind = k2 or ""
                if bad_kind:
                    _add(
                        "D001",
                        "error",
                        f'dataset.provider.kind "{bad_kind}" is not supported. Use one of: wikitext2 | hf_text | synthetic | local_jsonl.',
                        field="dataset.provider.kind",
                        hint="Use one of: wikitext2 | hf_text | synthetic | local_jsonl",
                    )
                    had_error = True
                # Schema-level validations per provider kind (support mapping-like)
                kind_val = None
                if isinstance(provider_cfg, dict):
                    kind_val = provider_cfg.get("kind")
                else:
                    kind_val = _pget(provider_cfg, "kind")
                kind = str(kind_val or "").strip()
                if kind == "local_jsonl":
                    p = None
                    if isinstance(provider_cfg, dict):
                        p = (
                            provider_cfg.get("file")
                            or provider_cfg.get("path")
                            or provider_cfg.get("data_files")
                        )
                    else:
                        p = (
                            _pget(provider_cfg, "file")
                            or _pget(provider_cfg, "path")
                            or _pget(provider_cfg, "data_files")
                        )
                    try:
                        from pathlib import Path as _P

                        exists = bool(p) and _P(str(p)).exists()
                    except Exception:
                        exists = False
                    if not exists:
                        _add(
                            "D011",
                            "error",
                            "local_jsonl: path does not exist",
                            field="dataset.provider.file",
                        )
                        had_error = True
                    tf = None
                    if isinstance(provider_cfg, dict):
                        tf = str(provider_cfg.get("text_field", "")).strip() or "text"
                    else:
                        tf = (
                            str(_pget(provider_cfg, "text_field") or "").strip()
                            or "text"
                        )
                    if not tf:
                        _add(
                            "D012",
                            "warning",
                            "local_jsonl: set dataset.field.text or map 'text' to your column",
                            field="dataset.provider.text_field",
                        )
                if kind == "hf_text":
                    tf2 = None
                    if isinstance(provider_cfg, dict):
                        tf2 = str(provider_cfg.get("text_field", "")).strip() or "text"
                    else:
                        tf2 = (
                            str(_pget(provider_cfg, "text_field") or "").strip()
                            or "text"
                        )
                    if not tf2:
                        _add(
                            "D012",
                            "warning",
                            "hf_text: set dataset.field.text or map 'text' to your column",
                            field="dataset.provider.text_field",
                        )
            except Exception:
                pass

            # Resolve adapter & provider
            adapter_name = (
                str(getattr(cfg.model, "adapter", "")).lower()
                if hasattr(cfg, "model")
                else ""
            )
            model_id_raw = (
                str(getattr(cfg.model, "id", "")) if hasattr(cfg, "model") else ""
            )
            model_profile = detect_model_profile(
                model_id=model_id_raw, adapter=adapter_name
            )
            metric_kind_resolved, provider_kind, _metric_opts = (
                _resolve_metric_and_provider(
                    cfg, model_profile, resolved_loss_type=None
                )
            )
            try:
                cfg_metric_kind = str(metric_kind_resolved)
            except Exception:
                cfg_metric_kind = cfg_metric_kind
            if not json_out:
                console.print(
                    f"  Metric: {metric_kind_resolved} · Provider: {provider_kind}"
                )

            # Resolve provider and tokenizer
            provider = get_provider(provider_kind)
            tokenizer, tok_hash = resolve_tokenizer(model_profile)
            if not json_out:
                console.print(
                    f"  Tokenizer: {tokenizer.__class__.__name__} · hash={tok_hash}"
                )

            # CUDA preflight (D002)
            try:
                import torch as _torch

                requested_device = None
                try:
                    requested_device = getattr(
                        getattr(cfg, "runner", object()), "device", None
                    )
                except Exception:
                    requested_device = None
                if requested_device is None:
                    try:
                        requested_device = getattr(
                            getattr(cfg, "model", object()), "device", None
                        )
                    except Exception:
                        requested_device = None
                req = str(requested_device or "").lower()
                if req.startswith("cuda") and not (
                    getattr(_torch, "cuda", None) and _torch.cuda.is_available()
                ):
                    _add(
                        "D002",
                        "error",
                        "CUDA requested but not available (runner.device=cuda). Resolve drivers / install CUDA PyTorch.",
                        field="runner.device",
                    )
                    had_error = True
            except Exception:
                pass

            # Determinism guard rails: warn when provider.workers>0 without deterministic_shards
            try:
                provider_cfg = None
                if hasattr(cfg.dataset, "provider"):
                    provider_cfg = cfg.dataset.provider
                # Accept mapping-shaped provider configs
                workers = None
                det = None
                if isinstance(provider_cfg, dict):
                    workers = provider_cfg.get("workers")
                    det = provider_cfg.get("deterministic_shards")
                else:
                    # Support InvarLockConfig's _Obj wrapper with dict-like get()
                    try:
                        workers = provider_cfg.get("workers", None)  # type: ignore[attr-defined]
                        det = provider_cfg.get("deterministic_shards", None)  # type: ignore[attr-defined]
                    except Exception:
                        workers = workers
                        det = det
                # Legacy style might place workers directly under dataset
                if workers is None and hasattr(cfg.dataset, "workers"):
                    workers = cfg.dataset.workers
                if det is None and hasattr(cfg.dataset, "deterministic_shards"):
                    det = cfg.dataset.deterministic_shards
                workers_val = int(workers) if workers is not None else 0
                det_flag = bool(det) if det is not None else False
                if workers_val > 0 and not det_flag:
                    # Print the canonical message and include a human-readable hint token
                    if not json_out:
                        console.print(
                            f"  [yellow]⚠️  {DETERMINISM_SHARDS_WARNING} (deterministic shards)[/yellow]"
                        )
            except Exception:
                # Best-effort linting only
                pass

            # Determinism hints (D004: low bootstrap reps)
            try:
                reps_val = None
                if hasattr(cfg, "eval") and hasattr(cfg.eval, "bootstrap"):
                    try:
                        reps_val = getattr(cfg.eval.bootstrap, "replicates", None)
                    except Exception:
                        reps_val = None
                if reps_val is not None:
                    try:
                        reps_val = int(reps_val)
                    except Exception:
                        reps_val = None
                if isinstance(reps_val, int) and reps_val < 200:
                    _add(
                        "D004",
                        "warning",
                        "bootstrap replicates (<200) may produce unstable CIs; increase reps or expect wider intervals.",
                        field="eval.bootstrap.replicates",
                    )
            except Exception:
                pass

            # Capacity estimation if available
            est = getattr(provider, "estimate_capacity", None)
            if callable(est):
                try:
                    seq_len = (
                        int(getattr(cfg.dataset, "seq_len", 512))
                        if hasattr(cfg, "dataset")
                        else 512
                    )
                    stride = (
                        int(getattr(cfg.dataset, "stride", seq_len // 2))
                        if hasattr(cfg, "dataset")
                        else seq_len // 2
                    )
                    preview_n = int(getattr(cfg.dataset, "preview_n", 0) or 0)
                    final_n = int(getattr(cfg.dataset, "final_n", 0) or 0)
                    cap = est(
                        tokenizer=tokenizer,
                        seq_len=seq_len,
                        stride=stride,
                        split=getattr(cfg.dataset, "split", "validation"),
                        target_total=preview_n + final_n,
                        fast_mode=True,
                    )
                    avail = cap.get("available_nonoverlap") or cap.get(
                        "candidate_limit"
                    )
                    console.print(
                        f"  Capacity: available={avail} · seq_len={seq_len} · stride={stride}"
                    )
                    if isinstance(avail, int) and (preview_n + final_n) > avail:
                        console.print(
                            "  [yellow]⚠️ Requested windows exceed provider capacity[/yellow]"
                        )
                    # Floors preview and capacity insufficiency (D007, D008)
                    try:
                        import math as _math

                        from invarlock.core.auto_tuning import get_tier_policies

                        use_tier = (tier or "balanced").lower()
                        tier_policies = get_tier_policies()
                        tier_defaults = tier_policies.get(
                            use_tier, tier_policies.get("balanced", {})
                        )
                        metrics_policy = (
                            tier_defaults.get("metrics", {})
                            if isinstance(tier_defaults, dict)
                            else {}
                        )
                        pm_policy = (
                            metrics_policy.get("pm_ratio", {})
                            if isinstance(metrics_policy, dict)
                            else {}
                        )
                        acc_policy = (
                            metrics_policy.get("accuracy", {})
                            if isinstance(metrics_policy, dict)
                            else {}
                        )
                        min_tokens = int(pm_policy.get("min_tokens", 0) or 0)
                        token_frac = float(
                            pm_policy.get("min_token_fraction", 0.0) or 0.0
                        )
                        min_examples = int(acc_policy.get("min_examples", 0) or 0)
                        ex_frac = float(
                            acc_policy.get("min_examples_fraction", 0.0) or 0.0
                        )
                        # Publish policy meta for JSON output
                        try:
                            global POLICY_META
                            POLICY_META = {
                                "tier": use_tier,
                                "floors": {
                                    "pm_ratio": {
                                        "min_tokens": min_tokens,
                                        "min_token_fraction": token_frac,
                                    },
                                    "accuracy": {
                                        "min_examples": min_examples,
                                        "min_examples_fraction": ex_frac,
                                    },
                                },
                            }
                        except Exception:
                            pass
                        tokens_avail = cap.get("tokens_available")
                        examples_avail = cap.get("examples_available")
                        eff_tokens = int(min_tokens)
                        eff_examples = int(min_examples)
                        if isinstance(tokens_avail, int | float) and token_frac > 0:
                            eff_tokens = max(
                                eff_tokens,
                                int(_math.ceil(float(tokens_avail) * token_frac)),
                            )
                        if isinstance(examples_avail, int | float) and ex_frac > 0:
                            eff_examples = max(
                                eff_examples,
                                int(_math.ceil(float(examples_avail) * ex_frac)),
                            )
                        if eff_tokens > 0 or eff_examples > 0:
                            _add(
                                "D007",
                                "note",
                                f"Floors: tokens >= {eff_tokens} (effective), examples >= {eff_examples} (effective)",
                                tokens_min=eff_tokens,
                                examples_min=eff_examples,
                            )
                        insufficient = False
                        if (
                            isinstance(tokens_avail, int | float)
                            and eff_tokens > 0
                            and tokens_avail < eff_tokens
                        ):
                            insufficient = True
                        if (
                            isinstance(examples_avail, int | float)
                            and eff_examples > 0
                            and examples_avail < eff_examples
                        ):
                            insufficient = True
                        if insufficient:
                            _add(
                                "D008",
                                "error",
                                f"Insufficient capacity: tokens_available={tokens_avail}, examples_available={examples_avail} below effective floors",
                            )
                            had_error = True
                    except Exception:
                        pass
                except Exception as _e:
                    console.print(
                        f"  [yellow]⚠️ Capacity estimation failed: {_e}[/yellow]"
                    )
            else:
                console.print(
                    "  [dim]Provider does not expose estimate_capacity()[/dim]"
                )

            # Baseline pairing sanity
            if baseline:
                try:
                    bpath = Path(baseline)
                    if bpath.exists():
                        bdata = _json.loads(bpath.read_text())
                        has_windows = isinstance(bdata.get("evaluation_windows"), dict)
                        console.print(
                            f"  Baseline windows: {'present' if has_windows else 'missing'}"
                        )
                        try:
                            prov = (
                                bdata.get("provenance", {})
                                if isinstance(bdata, dict)
                                else {}
                            )
                            if isinstance(prov, dict) and prov.get("split_fallback"):
                                _add(
                                    "D003",
                                    "warning",
                                    "dataset split fallback was used. Set dataset.provider.hf_dataset.split explicitly.",
                                )
                                if not json_out:
                                    console.print(
                                        f"  [yellow]⚠️  {DATASET_SPLIT_FALLBACK_WARNING}[/yellow]"
                                    )
                        except Exception:
                            pass
                    else:
                        console.print("  [yellow]⚠️ Baseline not found[/yellow]")
                except Exception as _e:
                    console.print(f"  [yellow]⚠️ Baseline check failed: {_e}[/yellow]")
        except Exception as e:
            console.print(f"  [yellow]⚠️ Preflight failed: {e}[/yellow]")

    # Baseline quick check for split fallback visibility (even without --config)
    try:
        if (baseline or baseline_report) and not config:
            from json import loads as _json_loads
            from pathlib import Path as _Path

            bpath = _Path(baseline or baseline_report)
            if bpath.exists():
                bdata = _json_loads(bpath.read_text())
                prov = bdata.get("provenance", {}) if isinstance(bdata, dict) else {}
                if isinstance(prov, dict) and prov.get("split_fallback"):
                    _add(
                        "D003",
                        "warning",
                        "dataset split fallback was used. Set dataset.provider.hf_dataset.split explicitly.",
                    )
                    if not json_out:
                        console.print(
                            f"  [yellow]⚠️  {DATASET_SPLIT_FALLBACK_WARNING}[/yellow]"
                        )
    except Exception:
        pass

    had_error = had_error or _cross_check_reports(
        baseline_report,
        subject_report,
        cfg_metric_kind=cfg_metric_kind,
        strict=bool(strict),
        profile=profile,
        json_out=json_out,
        console=console,
        add_fn=_add,
    )

    # D013: Tiny relax (dev) active — note only
    try:
        tiny_env = str(_os.environ.get("INVARLOCK_TINY_RELAX", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        tiny_env = False
    tiny_cert = False
    try:
        # Best-effort: detect from reports when provided
        import json as _json_d13
        from pathlib import Path as _Path_d13

        def _readsafe(p):
            try:
                return _json_d13.loads(_Path_d13(p).read_text()) if p else None
            except Exception:
                return None

        sb = _readsafe(subject_report) if subject_report else None
        bb = _readsafe(baseline_report) if baseline_report else None
        tiny_cert = bool(
            ((sb or {}).get("auto", {}) or {}).get("tiny_relax")
            or ((bb or {}).get("auto", {}) or {}).get("tiny_relax")
        )
    except Exception:
        tiny_cert = False
    if tiny_env or tiny_cert:
        _add(
            "D013",
            "note",
            "tiny relax (dev) active; gates widened and drift/overhead may be informational.",
            field="auto.tiny_relax",
        )

    # Check registry status
    try:
        from invarlock.core.registry import get_registry

        from .plugins import _check_plugin_extras

        if not json_out:
            console.print("\n🔌 Plugin Registry")
        registry = get_registry()
        if not json_out:
            console.print(f"  Adapters: {len(registry.list_adapters())}")
            console.print(f"  Edits: {len(registry.list_edits())}")
            console.print(f"  Guards: {len(registry.list_guards())}")
        # Use module-level _os (avoid shadowing earlier uses)
        if _os.getenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "").strip() == "1":
            _add(
                "D006",
                "note",
                "Plugin discovery disabled; doctor will not check optional adapters.",
            )

        # Detail adapters with Origin/Mode/Backend/Version table
        def _gather_adapter_rows() -> list[dict]:
            names = registry.list_adapters()
            try:
                import torch as _t

                has_cuda = bool(getattr(_t, "cuda", None) and _t.cuda.is_available())
            except Exception:
                has_cuda = False
            is_linux = _platform.system().lower() == "linux"

            rows: list[dict] = []
            for n in names:
                info = registry.get_plugin_info(n, "adapters")
                module = str(info.get("module") or "")
                support = (
                    "auto"
                    if module.startswith("invarlock.adapters")
                    and n in {"hf_causal_auto", "hf_mlm_auto"}
                    else (
                        "core"
                        if module.startswith("invarlock.adapters")
                        else "optional"
                    )
                )
                origin = "core" if support in {"core", "auto"} else "plugin"
                mode = "auto-matcher" if support == "auto" else "adapter"

                backend, version = None, None
                status, enable = "ready", ""

                # Heuristic backend mapping without heavy imports
                if n in {
                    "hf_gpt2",
                    "hf_bert",
                    "hf_llama",
                    "hf_causal_auto",
                    "hf_mlm_auto",
                }:
                    # Transformers-based
                    backend = "transformers"
                    try:
                        import transformers as _tf  # type: ignore

                        version = getattr(_tf, "__version__", None)
                    except Exception:
                        version = None
                elif n == "hf_gptq":
                    backend = "auto-gptq"
                elif n == "hf_awq":
                    backend = "autoawq"
                elif n == "hf_bnb":
                    backend = "bitsandbytes"

                # Presence and platform gating
                if support == "optional":
                    # Check install presence
                    present = (
                        importlib.util.find_spec((backend or "").replace("-", "_"))
                        is not None
                        if backend
                        else False
                    )
                    if not present:
                        status = "needs_extra"
                        hint = {
                            "hf_gptq": "invarlock[gptq]",
                            "hf_awq": "invarlock[awq]",
                            "hf_bnb": "invarlock[gpu]",
                        }.get(n)
                        if hint:
                            enable = f"pip install '{hint}'"
                # Special-case: hf_onnx is a core adapter but requires Optimum/ONNXRuntime
                if n == "hf_onnx":
                    backend = backend or "onnxruntime"
                    present = (
                        importlib.util.find_spec("optimum.onnxruntime") is not None
                        or importlib.util.find_spec("onnxruntime") is not None
                    )
                    if not present:
                        status = "needs_extra"
                        enable = "pip install 'invarlock[onnx]'"
                # Platform checks
                if backend in {"auto-gptq", "autoawq"} and not is_linux:
                    status = "unsupported"
                    enable = "Linux-only"
                if backend == "bitsandbytes" and not has_cuda:
                    status = "unsupported"
                    enable = "Requires CUDA"

                rows.append(
                    {
                        "name": n,
                        "origin": origin,
                        "mode": mode,
                        "backend": backend,
                        "version": version,
                        "status": status,
                        "enable": enable,
                    }
                )
            return rows

        def _fmt_backend_ver(
            backend: str | None, version: str | None
        ) -> tuple[str, str]:
            b = backend or "—"
            v = f"=={version}" if backend and version else "—"
            return b, v

        # Build adapter rows; gracefully handle optional Optimum import errors by
        # falling back to a lightweight rows helper that only probes availability.
        try:
            all_rows = _gather_adapter_rows()
        except Exception as _adapter_exc:
            # Known benign case: optional Optimum/ONNXRuntime missing on host
            if "optimum" in str(_adapter_exc).lower():
                try:
                    from invarlock.cli.doctor_helpers import (
                        get_adapter_rows as _rows_fallback,
                    )

                    all_rows = _rows_fallback()
                except Exception:
                    raise  # re-raise if fallback also fails
            else:
                raise
        if all_rows:
            # Counts over full set
            total = len(all_rows)
            ready = sum(1 for r in all_rows if r["status"] == "ready")
            need = sum(1 for r in all_rows if r["status"] == "needs_extra")
            unsupported = sum(1 for r in all_rows if r["status"] == "unsupported")
            auto = sum(1 for r in all_rows if r["mode"] == "auto-matcher")
            # Hide unsupported rows in the display
            rows = [r for r in all_rows if r["status"] != "unsupported"]
            table = Table(
                title=f"Adapters — total: {total} · ready: {ready} · auto: {auto} · missing-extras: {need} · unsupported: {unsupported}"
            )
            table.add_column("Adapter", style="cyan")
            table.add_column("Origin", style="dim")
            table.add_column("Mode", style="dim")
            table.add_column("Backend", style="magenta")
            table.add_column("Version", style="magenta")
            table.add_column("Status / Action", style="green")
            for r in rows:
                backend_disp, ver_disp = _fmt_backend_ver(r["backend"], r["version"])
                if r["mode"] == "auto-matcher":
                    status_disp = "🧩 Auto (selects best hf_* adapter)"
                elif r["status"] == "ready":
                    status_disp = "✅ Ready"
                elif r["status"] == "needs_extra":
                    status_disp = (
                        f"⛔ Needs extra → {r['enable']}"
                        if r["enable"]
                        else "⛔ Needs extra"
                    )
                else:
                    status_disp = r["status"]
                table.add_row(
                    r["name"],
                    r["origin"].capitalize(),
                    "Auto‑matcher" if r["mode"] == "auto-matcher" else "Adapter",
                    backend_disp,
                    ver_disp,
                    status_disp,
                )
            console.print(table)

        # Guards table
        def _gather_generic_rows(kind: str) -> list[dict]:
            names = (
                registry.list_guards() if kind == "guards" else registry.list_edits()
            )
            rows: list[dict] = []
            for n in names:
                info = registry.get_plugin_info(n, kind)
                module = str(info.get("module") or "")
                origin = "core" if module.startswith(f"invarlock.{kind}") else "plugin"
                mode = "guard" if kind == "guards" else "edit"
                # Extras
                status = "ready"
                enable = ""
                try:
                    extras = _check_plugin_extras(n, kind)
                except Exception:
                    extras = ""
                if (
                    isinstance(extras, str)
                    and extras.startswith("⚠️")
                    and "missing" in extras
                ):
                    status = "needs_extra"
                    hint = extras.split("missing", 1)[-1].strip()
                    if hint:
                        enable = f"pip install '{hint}'"
                rows.append(
                    {
                        "name": n,
                        "origin": origin,
                        "mode": mode,
                        "backend": None,
                        "version": None,
                        "status": status,
                        "enable": enable,
                    }
                )
            return rows

        for kind, title in (("guards", "Guards"), ("edits", "Edits")):
            grows = _gather_generic_rows(kind)
            if grows:
                total = len(grows)
                ready = sum(1 for r in grows if r["status"] == "ready")
                need = sum(1 for r in grows if r["status"] == "needs_extra")
                table = Table(
                    title=f"{title} — total: {total} · ready: {ready} · missing-extras: {need}"
                )
                table.add_column("Name", style="cyan")
                table.add_column("Origin", style="dim")
                table.add_column("Mode", style="dim")
                table.add_column("Backend", style="magenta")
                table.add_column("Version", style="magenta")
                table.add_column("Status / Action", style="green")
                for r in grows:
                    b, v = _fmt_backend_ver(r["backend"], r["version"])

                    status_disp = (
                        "✅ Ready"
                        if r["status"] == "ready"
                        else (
                            f"⛔ Needs extra → {r['enable']}"
                            if r["enable"]
                            else "⛔ Needs extra"
                        )
                    )
                    table.add_row(
                        r["name"],
                        r["origin"].capitalize(),
                        ("Guard" if r["mode"] == "guard" else "Edit"),
                        b,
                        v,
                        status_disp,
                    )
                console.print(table)

        # Datasets summary (best effort; non-fatal)
        try:
            from invarlock.eval.data import list_providers  # type: ignore

            providers = sorted(list_providers())
            if providers:
                dtable = Table(title="Datasets")
                dtable.add_column("Provider", style="cyan")
                dtable.add_column("Network", style="dim")
                dtable.add_column("Status", style="green")
                dtable.add_column("Params", style="dim")
                from invarlock.cli.constants import (
                    PROVIDER_NETWORK as provider_network,
                )
                from invarlock.cli.constants import (
                    PROVIDER_PARAMS as provider_params,
                )

                def _net_label(name: str) -> str:
                    val = (provider_network.get(name, "") or "").lower()
                    if val == "cache":
                        return "Cache/Net"
                    if val == "yes":
                        return "Yes"
                    if val == "no":
                        return "No"
                    return "Unknown"

                for pname in providers:
                    dtable.add_row(
                        pname,
                        _net_label(pname),
                        "✓ Available",
                        provider_params.get(pname, "-"),
                    )
                console.print(dtable)
        except Exception:
            pass

        if not json_out:
            console.print(
                "[dim]Legend: ✅ Ready = usable now · 🧩 Auto‑matcher = picks an adapter for you[/dim]"
            )
            console.print(
                "[dim]Hints: use --json · filter with --only ready|core|plugin|auto|unsupported[/dim]"
            )
    except Exception as e:
        # Gracefully handle missing optional Optimum stack
        if "optimum" in str(e).lower():
            if not json_out:
                console.print(
                    "  [yellow]⚠️  Optional Optimum/ONNXRuntime missing; hf_onnx will be shown as needs_extra[/yellow]"
                )
            # Do not mark overall health as failed for optional extras
        else:
            if not json_out:
                console.print(f"  [red]❌ Registry error: {e}[/red]")
            health_status = False

    # Final status / JSON output
    exit_code = 0 if (health_status and not had_error) else 1
    if json_out:
        import json as _json_out

        # Sort findings deterministically by severity then code
        _order = {"error": 0, "warning": 1, "note": 2}
        try:
            findings.sort(
                key=lambda f: (_order.get(f.get("severity"), 9), f.get("code", "Z999"))
            )
        except Exception:
            pass
        result_obj = {
            "format_version": DOCTOR_FORMAT_VERSION,
            "summary": {
                "errors": sum(1 for f in findings if f.get("severity") == "error"),
                "warnings": sum(1 for f in findings if f.get("severity") == "warning"),
                "notes": sum(1 for f in findings if f.get("severity") == "note"),
            },
            "policy": POLICY_META
            if "POLICY_META" in globals()
            else {"tier": (tier or "balanced").lower()},
            "findings": findings,
            "resolution": {"exit_code": exit_code},
        }
        typer.echo(_json_out.dumps(result_obj))
        raise typer.Exit(exit_code)
    else:
        console.print("\n" + "=" * 50)
        if exit_code == 0:
            console.print(
                "[green]✅ InvarLock installation is healthy (exit code 0)[/green]"
            )
        else:
            console.print("[red]❌ InvarLock installation has issues[/red]")
            console.print(
                "Run: pip install invarlock[all] to install missing dependencies"
            )
        sys.exit(exit_code)
