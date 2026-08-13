#audit_model_feature_order.py

from inference.match_predictor import MatchPredictor


def main() -> None:
    predictor = MatchPredictor()
    model = predictor.model

    print("Model type:", type(model))

    if hasattr(model, "feature_names_in_"):
        print()
        print("feature_names_in_")
        print("-----------------")
        for i, name in enumerate(model.feature_names_in_):
            print(f"{i:2}  {name}")
    else:
        print("Model does not expose feature_names_in_")

    if hasattr(model, "n_features_in_"):
        print()
        print("n_features_in_:", model.n_features_in_)


if __name__ == "__main__":
    main()