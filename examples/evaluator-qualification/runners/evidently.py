"""Execute Evidently's upstream per-row ExactMatch descriptor."""

import pandas as pd
from evidently import DataDefinition, Dataset
from evidently.descriptors import ExactMatch
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    source = pd.DataFrame(
        [
            {
                "output": case["output"],
                "record_id": case["record_id"],
                "reference": case["reference"],
            }
            for case in cases
        ]
    )
    dataset = Dataset.from_pandas(
        source,
        data_definition=DataDefinition(),
        descriptors=[
            ExactMatch(
                columns=["output", "reference"],
                alias="exact_match",
            )
        ],
    )
    scored = dataset.as_dataframe()
    by_id = {
        str(row["record_id"]): bool(row["exact_match"]) for _, row in scored.iterrows()
    }
    scores = [1.0 if by_id[case["record_id"]] else 0.0 for case in cases]
    details = [{"exact_match": bool(score)} for score in scores]
    finish_deterministic(
        args=args,
        entrypoint="evidently.Dataset.from_pandas/Evidently ExactMatch",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
