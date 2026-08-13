# train_random_forest.py

from sklearn.ensemble import RandomForestClassifier

from shared.config import ML_OUTPUT_DIR
from shared.dataset import prepare_train_test_data
from shared.experiment import run_experiment
from shared.feature_sets import FEATURE_SETS
from shared.model_config import RANDOM_FOREST_CONFIG
from shared.results_summary import save_summary, print_summary_table


MODEL_NAME = "RandomForest"


def train_random_forest_for_feature_set(feature_set_name, features):
    print(f"\nTraining Random Forest using feature set: {feature_set_name}")

    X, y, X_train, X_test, y_train, y_test, labels, metadata = (
        prepare_train_test_data(features)
    )

    print(f"Rows available: {metadata['rows_used']}")
    print()

    print("Class distribution:")
    class_counts = (
        y_train.value_counts()
        .add(y_test.value_counts(), fill_value=0)
        .sort_index()
    )

    total_rows = metadata["rows_used"]

    for label, count in class_counts.items():
        pct = count / total_rows * 100
        print(f"  {label}: {int(count)} ({pct:.1f}%)")

    print()

    print(f"Training rows : {metadata['train_rows']}")
    print(f"Testing rows  : {metadata['test_rows']}")
    print(f"Features      : {len(features)}")
    print()

    model = RandomForestClassifier(**RANDOM_FOREST_CONFIG)

    metadata.update(
        {
            "model": MODEL_NAME,
            "feature_set": feature_set_name,
            "features": features,
        }
    )

    return run_experiment(
        model_name=MODEL_NAME,
        model=model,
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        labels=labels,
        feature_set_name=feature_set_name,
        features=features,
        output_dir=ML_OUTPUT_DIR / "random_forest",
        metadata=metadata,
    )


def main():
    summary_rows = []

    for feature_set_name, features in FEATURE_SETS.items():
        summary_row = train_random_forest_for_feature_set(
            feature_set_name,
            features,
        )
        summary_rows.append(summary_row)

    summary_path = ML_OUTPUT_DIR / "random_forest_summary.csv"
    summary_df = save_summary(summary_rows, summary_path)

    print_summary_table(summary_df)
    print()
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()