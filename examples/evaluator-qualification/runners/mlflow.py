"""Execute MLflow model evaluation and retain aggregate-only observations."""

import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from runner_support import arguments, finish_observation, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    data = pd.DataFrame(
        {
            "prediction": [case["output"] for case in cases],
            "target": [case["reference"] for case in cases],
        }
    )
    with tempfile.TemporaryDirectory(prefix="invarlock-mlflow-") as temporary:
        root = Path(temporary)
        os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
        mlflow.set_tracking_uri(f"sqlite:///{root / 'tracking.db'}")
        client = mlflow.MlflowClient()
        experiment_id = client.create_experiment(
            "invarlock-evaluator-qualification",
            artifact_location=(root / "artifacts").as_uri(),
        )
        with mlflow.start_run(experiment_id=experiment_id):
            result = mlflow.models.evaluate(
                model=None,
                data=data,
                targets="target",
                predictions="prediction",
                model_type="classifier",
            )
    accuracy = float(result.metrics["accuracy_score"])
    finish_observation(
        args=args,
        entrypoint="mlflow.models.evaluate",
        summary_kind="aggregate_metrics",
        summary_data={
            "accuracy_score": accuracy,
            "evaluated_records": len(cases),
        },
    )


if __name__ == "__main__":
    main()
