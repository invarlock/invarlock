# Model Family Catalog

## Overview

The model family catalog connects InvarLock's maintained evaluation lanes with
the broader adapter and loader coverage available in the repository.

Three contracts provide complementary views:

| Contract | Purpose |
| --- | --- |
| `contracts/evidence_catalog_v1.json` | Exact models, adapters, presets, inputs, and required artifacts for maintained evaluation lanes. |
| `contracts/support_matrix.json` | Support tier and current evidence status for each lane. |
| `contracts/model_family_catalog.json` | Model-family implementation coverage and representative checkpoints. |

## Maintained evaluation lanes

The repository currently defines 39 maintained lanes covering causal language
models, masked language models, seq2seq models, dense and MoE architectures,
and image-text evaluation. Their adapters and presets are available today.

Current-contract evidence is **Available** for 23 lanes. The other 16 lanes
retain **Evidence not yet created** until their artifacts are published.

Use the [Support Matrix](../README.md#support-matrix) for the complete table.

## Implementation coverage

`implemented_coverage` in `contracts/model_family_catalog.json` records model
families that share maintained loader, profile, or adapter paths. This makes it
easy to discover compatible checkpoints and identify where a dedicated
catalog lane would add value.

Coverage states include:

| State | Meaning |
| --- | --- |
| `profile_first_class` | The family has dedicated profile recognition. |
| `profile_shared_alias` | The family uses a maintained shared profile path. |
| `auto_or_loader_only` | Adapter-auto or loader support is available. |
| `loader_only` | The checkpoint can use an existing loader path. |

## Adding a maintained lane

A model family is ready for the maintained catalog when it has:

1. explicit adapter or profile recognition;
2. an included evaluation preset;
3. an included calibration configuration;
4. targeted tests;
5. a working CLI smoke path;
6. a catalog entry with immutable input and artifact requirements.

Evidence publication then records the completed evaluation and verification
artifacts without changing the lane definition.

## Related documentation

- [Model Adapters](model-adapters.md)
- [Public Contracts](contracts.md)
- [Tier Policy Tuning](calibration.md)
- [Evidence Catalog](../user-guide/public-evidence-walkthrough.md)
