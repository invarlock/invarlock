# Support and Questions

Thank you for using InvarLock! This document explains where to ask questions, how to report issues, and what response expectations to have.

## Where to Ask What

- **Usage questions, how‑to, and design discussions**
  - Use **GitHub Discussions** (if enabled) or start a thread in the Q&A/Ideas categories.
  - Example topics: “How do I run certify with my own checkpoints?”, “What profile should I use for CPU‑only runs?”.

- **Bugs and regressions**
  - Open a **GitHub Issue** using the Bug Report template.
  - Include:
    - InvarLock version (`invarlock --version`),
    - Python version and OS,
    - Exact command you ran (or minimal script),
    - Relevant config file(s) or a minimal redacted snippet,
    - Full error message and traceback.

- **Feature requests and enhancements**
  - Open a **Feature Request** issue to propose new functionality, adapters, or docs improvements.
  - Describe the use case, not just the feature (what problem are you trying to solve?).

- **Security and vulnerability reports**
  - Follow the process in [`SECURITY.md`](SECURITY.md).
  - Do **not** file public issues for suspected vulnerabilities.

## Response Expectations

- This project is maintained on a **best‑effort** basis.
- There is **no guaranteed SLA** for responses or fixes.
- We periodically review new issues and discussions and try to prioritise:
  - Clearly‑reproducible bugs, especially recent regressions,
  - Security reports (via `SECURITY.md`),
  - Documentation gaps that affect many users.
- Response times can vary significantly, especially around holidays, conferences, and major releases.

If your issue is blocking critical work, please:

- Provide a minimal reproducible example,
- Confirm whether it reproduces on the latest released version,
- Add any relevant logs from `invarlock doctor` or `invarlock verify`.

## Before Filing an Issue

Please check the following first:

1. **Search existing issues and discussions** for similar reports or known limitations.
2. **Check the docs**:
   - Quickstart and getting started guides under `docs/user-guide/`,
   - CLI reference in `docs/reference/cli.md`.
3. **Run `invarlock doctor`** on your config to catch common misconfigurations.
4. If the issue involves external dependencies (PyTorch, transformers, datasets, etc.), confirm you can reproduce it with a minimal InvarLock‑centric example.

Issues that do not follow the templates, or that are clearly out of scope (e.g., generic deep learning questions), may be closed with a reference to this document.

## Out of Scope

The issue tracker is not the right place for:

- General PyTorch or transformers debugging help,
- Model training or fine‑tuning advice unrelated to InvarLock’s certification flow,
- Broad ML consultations or architecture reviews,
- Vendor‑specific support for hardware or cloud environments.

For these topics, please use community channels (e.g., PyTorch/transformers forums or general ML communities).
