# Native export fixtures

The three batch fixtures below retain decision-relevant native fields from locally executed
upstream evaluation runs on 2026-09-05. Each run contains 40 synthetic cases;
no model service was called. They test actual export interoperability and do
not demonstrate model performance, customer use or representative data.

| Fixture | Executed source | Capture |
| --- | --- | --- |
| `inspect-0.3.254.json` | `inspect-ai==0.3.254`, `inspect_ai.eval` | A Task with 40 Samples, a solver assigning captured `ModelOutput` text, and the upstream `match` scorer; JSON EvalLog |
| `lm-eval-0.4.12.jsonl` | `lm-eval==0.4.12`, `lm_eval.evaluator.evaluate` | A local ConfigurableTask dataset, a deterministic LM returning captured text, and generation sample logging |
| `promptfoo-0.121.19.jsonl` | `promptfoo@0.121.19`, `promptfoo eval` | Forty local tests, the `echo` provider, an equals assertion, and the native full JSONL export |

Inspect retains version, status and sample input, target, output, scores, tags
and epoch. LM Evaluation Harness retains document, ID, target, arguments,
filtered response and its match score. Promptfoo retains prompt, stable indexes,
test case, response, score, latency and cost. Other run metadata, host paths and
unused logging fields are omitted. The retained field values were not rewritten.
Native import computes the digest of these exact fixture bytes, not the larger
original log. Tests also exercise malformed and unsupported shapes separately.

`promptfoo-0.121.19-assertion-failure.json` is a separate single-row capture from
a real local model call through Promptfoo 0.121.19 on 2026-09-06. The model returned
text successfully but failed the native `equals` assertion. It retains the
original assertion, output, score, latency and `failureReason: 1` fields while
omitting volatile HTTP metadata and unrelated logging fields. It demonstrates
the export's distinction between a wrong answer and an execution error; one
response does not establish model quality or representative performance.
