# Source attribution and changes

This reference contains frozen model answers and task prompts derived from
SQuAD 2.0 (Pranav Rajpurkar, Robin Jia and Percy Liang, *Know What You Don't
Know: Unanswerable Questions for SQuAD*, ACL 2018) and Schema-Guided Dialogue
(Abhinav Rastogi, Xiaoxue Zang, Srinivas Sunkara, Raghav Gupta and Pranav Khaitan,
*Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue
Dataset*, AAAI 2020). Dataset source URLs, revisions, sizes and hashes are retained
in source-attribution.json; unchanged publisher README and license files are
included beside this notice.

SGD source revision: e852981ae34990f4358979625854259302feaa78. SGD-derived data
is distributed under CC BY-SA 4.0; preserve the included license and attribution.
SQuAD includes Wikipedia-derived context. Preserve SQuAD and Wikipedia attribution
and applicable source terms. SQuAD-SOFTWARE-LICENSE.txt is the publisher's MIT
software license; it does not relicense all dataset text as MIT. Dataset material
is not relicensed under InvarLock's Apache-2.0 software license.

Changes: source dialogue schemas/current utterances and Wikipedia context/questions
were formatted as task prompts; K2 Horizon 32B generated frozen A/B answers.
This reference selects English QA and extraction source clusters with a published
hash rule, extracts the original single user-message text without repair, preserves
raw selected answers, metadata, original metrics and capture provenance separately,
and constructs new score-free evaluation records. It does not rerun generation,
change native scores, assert native runtime qualification, or imply endorsement by
model or dataset authors.
