# K2 Horizon 32B frozen-answer reference

This deterministic archive retains the outcome-blind judge-study inputs and
source mappings for English grounded QA and slot extraction. It contains no
new judge outcomes. Pilot review uses 40 cases per workflow. Final membership
is frozen at 422 QA clusters and 1,288 extraction clusters; final-validation
review uses 80 cases per workflow, balanced across its two strata.

**Final plans are candidates pending pilot rubric review.** Only pilot plans
are executable measurement-plan documents. Final plans and policies remain
inside an explicit candidate wrapper until the pilot review confirms or revises
the rubric. Changing the rubric must preserve frozen final membership.

From the source checkout with InvarLock installed, validate without model calls:

```bash
python examples/judge_measurements_reference.py validate \
  --bundle examples/judge-measurements/references/k2-32b/reference.zip \
  --expected-sha256 cd403375fe0e4dcfb40978e10c60e7bb0a126f1dd5b54de71a1818ae27b65002
```

[archive.json](archive.json) pins the ZIP transport and internal reference
manifest. Obtain those pins from an independent trusted copy of the repository.
Default validation reconstructs retained selection, mappings, contracts and
review sheets from the archive alone. Optional campaign-root validation also
rechecks every original source block, endpoint, protocol and planned QA case.
Internal consistency does not independently authenticate omitted original files.

For selection rules, exact frozen protocol and the build command, see the
[K2 reference guide](../../K2-REFERENCE.md). Give human reviewers only the
appropriate rubric-development or final-validation sheet from the archive,
keeping source roles and native scores withheld until their judgments are frozen.

Preserve [source attribution](ATTRIBUTION.md), the
[SGD data license](SGD-LICENSE.txt), and the accurately scoped
[SQuAD software license](SQuAD-SOFTWARE-LICENSE.txt). Complete original publisher
README files and pinned dataset source metadata are inside the archive.
