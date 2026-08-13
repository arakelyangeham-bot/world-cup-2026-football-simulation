# models/neural_net/train_neural_net.py

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.config import ML_OUTPUT_DIR
from shared.dataset import prepare_train_test_data
from shared.experiment import run_experiment
from shared.experiment_registry import register_experiments
from shared.feature_sets import FEATURE_SETS
from shared.label_encoding import encode_labels
from shared.model_config import MLP_CONFIG
from shared.results_summary import save_summary, print_summary_table


MODEL_NAME = "NeuralNet"


def train_neural_net_for_feature_set(feature_set_name, features):
    print(f"\nTraining Neural Net using feature set: {feature_set_name}")

    X, y, X_train, X_test, y_train, y_test, labels, metadata = (
        prepare_train_test_data(features)
    )

    (
        y_encoded,
        y_train_encoded,
        y_test_encoded,
        encoded_labels,
        label_mapping,
        _,
    ) = encode_labels(y, y_train, y_test)

    print(f"Rows available: {metadata['rows_used']}")
    print()
    print(f"Training rows : {metadata['train_rows']}")
    print(f"Testing rows  : {metadata['test_rows']}")
    print(f"Features      : {len(features)}")
    print()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**MLP_CONFIG)),
        ]
    )

    metadata.update(
        {
            "model": MODEL_NAME,
            "feature_set": feature_set_name,
            "features": features,
            "label_mapping": label_mapping,
        }
    )

    return run_experiment(
        model_name=MODEL_NAME,
        model=model,
        X=X,
        y=y_encoded,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train_encoded,
        y_test=y_test_encoded,
        labels=encoded_labels,
        feature_set_name=feature_set_name,
        features=features,
        output_dir=ML_OUTPUT_DIR / "neural_net",
        metadata=metadata,
    )


def main():
    summary_rows = []

    for feature_set_name, features in FEATURE_SETS.items():
        summary_rows.append(
            train_neural_net_for_feature_set(feature_set_name, features)
        )

    summary_path = ML_OUTPUT_DIR / "neural_net_summary.csv"
    summary_df = save_summary(summary_rows, summary_path)

    registry_path = ML_OUTPUT_DIR / "experiment_results.csv"
    registry_df = register_experiments(summary_df, registry_path)

    print_summary_table(summary_df)
    print()
    print(f"Saved summary: {summary_path}")

    print()
    print("Updated experiment registry:")
    print_summary_table(registry_df)
    print()
    print(f"Saved registry: {registry_path}")


if __name__ == "__main__":
    main()