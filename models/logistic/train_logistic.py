# train_logistic.py

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from shared.config import (
    ML_OUTPUT_DIR,
    RANDOM_STATE
)

from shared.model_config import LOGISTIC_CONFIG

from shared.feature_sets import FEATURE_SETS

from shared.experiment import run_experiment
from shared.results_summary import save_summary, print_summary_table
from shared.dataset import prepare_train_test_data
from shared.experiment_registry import register_experiments



def validate_columns(df, features):
    missing = [col for col in features + [TARGET_COLUMN] if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def train_logistic_for_feature_set(feature_set_name, features):
    print(f"\nTraining Logistic Regression using feature set: {feature_set_name}")

    X, y, X_train, X_test, y_train, y_test, labels, metadata = prepare_train_test_data(
        features
    )

    print(f"Rows available: {metadata['rows_used']}")
    print()

    print("Class distribution:")
    class_counts = y_train.value_counts().add(y_test.value_counts(), fill_value=0).sort_index()

    total_rows = metadata["rows_used"]

    for label, count in class_counts.items():
        pct = count / total_rows * 100
        print(f"  {label}: {int(count)} ({pct:.1f}%)")

    print()

    print(f"Training rows : {metadata['train_rows']}")
    print(f"Testing rows  : {metadata['test_rows']}")
    print(f"Features      : {len(features)}")
    print()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(**LOGISTIC_CONFIG),
            ),
        ]
    )

    metadata.update(
        {
            "model": "LogisticRegression",
            "feature_set": feature_set_name,
            "features": features,
        }
    )

    return run_experiment(
        model_name="LogisticRegression",
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
        output_dir=ML_OUTPUT_DIR / "logistic",
        metadata=metadata,
    )


def main():
    summary_rows = []

    print("=" * 70)
    print("Feature Set Summary")
    print("=" * 70)

    for name, features in FEATURE_SETS.items():
        print(f"{name}: {len(features)} features")

        if name == "v2":
            extra = sorted(set(features) - set(FEATURE_SETS["v1"]))
            if extra:
                print(f"  New since v1: {extra}")

    print()

    for feature_set_name, features in FEATURE_SETS.items():
        summary_row = train_logistic_for_feature_set(feature_set_name, features)
        summary_rows.append(summary_row)

    summary_path = ML_OUTPUT_DIR / "model_summary.csv"
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