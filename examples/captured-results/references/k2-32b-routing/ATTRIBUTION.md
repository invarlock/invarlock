# Source attribution and changes

The routing prompts contain service schemas and dialogue excerpts from the
Schema-Guided Dialogue dataset (SGD), published by Google Research. The pinned
upstream revision is `e852981ae34990f4358979625854259302feaa78`.
The upstream [README and attribution](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/README.md)
identifies the dataset license as
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
The unchanged license text is included in [SGD-LICENSE.txt](SGD-LICENSE.txt).

Citation: Abhinav Rastogi, Xiaoxue Zang, Srinivas Sunkara, Raghav Gupta and
Pranav Khaitan, “Towards Scalable Multi-Domain Conversational Agents: The
Schema-Guided Dialogue Dataset,” Proceedings of the AAAI Conference on
Artificial Intelligence, volume 34, number 05, pages 8689–8696.

Changes from SGD: a fixed schedule selects dialogue histories and named-service
schemas from 2,183 training, 932 development and 885 test examples, formats them as routing prompts, and adds alternative
prompt instructions, model-generated intent labels, recorded scores, provenance
and slice metadata. The current evidence representation also makes the format
and candidate-to-subject vocabulary changes described in the [reference](README.md).
No endorsement by the dataset authors is implied.

The SGD-derived prompt and record material is provided under CC BY-SA 4.0.
It does not acquire the software's Apache-2.0 license. Preserve this attribution,
the license and change notice when sharing that material. Original InvarLock
helper code remains under the repository's Apache-2.0 license.

This package contains only the routing workflow's SGD-derived material and short
model outputs. It contains no model weights, tokenizer files, financial-report
extracts, question-answering articles or database payloads from other campaign
workflows.
