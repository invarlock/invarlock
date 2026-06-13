# Qwen3 30B-A3B Published-Basis Evidence Pack

This pack contains the strict release-profile no-op preservation evidence for `Qwen/Qwen3-30B-A3B-Instruct-2507` on public WikiText-103.

Scope: this is a verifier fixture for InvarLock guard/report behavior. It is not a benchmark-quality, fine-tuning, compression, deployment, or MoE routing-quality claim. The successful run used all eight H100 80GB GPUs on `root@31.56.109.46`; a restricted four-GPU diagnostic lane failed during RMT/forward execution and is therefore not used as publication evidence.

Verify with:

```bash
scripts/evidence_packs/verify_pack.sh --pack <pack-dir> --strict
```
