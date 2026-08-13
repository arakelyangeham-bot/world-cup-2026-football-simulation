# train_xgboost.py

from xgboost import XGBClassifier

from shared.config import ML_OUTPUT_DIR
from shared.dataset import prepare_train_test_data
from shared.experiment import run_experiment
from shared.experiment_registry import register_experiments
from shared.feature_sets import FEATURE_SETS
from shared.model_config import XGBOOST_CONFIG
from shared.results_summary import save_summary, print_summary_table
from sklearn.preprocessing import LabelEncoder


MODEL_NAME = "XGBoost"


def train_xgboost_for_feature_set(feature_set_name, features):
    print(f"\nTraining XGBoost using feature set: {feature_set_name}")

    X, y, X_train, X_test, y_train, y_test, labels, metadata = (
        prepare_train_test_data(features)
    )

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    encoded_labels = list(range(len(label_encoder.classes_)))

    print(f"Rows available: {metadata['rows_used']}")
    print()
    print(f"Training rows : {metadata['train_rows']}")
    print(f"Testing rows  : {metadata['test_rows']}")
    print(f"Features      : {len(features)}")
    print()

    model = XGBClassifier(**XGBOOST_CONFIG)

    metadata.update(
        {
            "model": MODEL_NAME,
            "feature_set": feature_set_name,
            "features": features,
            "label_mapping": {
                int(i): label
                for i, label in enumerate(label_encoder.classes_)
            },
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
        output_dir=ML_OUTPUT_DIR / "xgboost",
        metadata=metadata,
    )


def main():
    summary_rows = []

    for feature_set_name, features in FEATURE_SETS.items():
        summary_rows.append(
            train_xgboost_for_feature_set(feature_set_name, features)
        )

    summary_path = ML_OUTPUT_DIR / "xgboost_summary.csv"
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