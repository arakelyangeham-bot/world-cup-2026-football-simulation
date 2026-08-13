#audit_negative_binomial_support.py

def main() -> None:
    print("Negative Binomial Support Audit")
    print("-------------------------------")

    try:
        import statsmodels.api as sm
        print("statsmodels: OK")
        print("statsmodels version:", sm.__version__)
    except Exception as exc:
        print("statsmodels: MISSING")
        print(exc)

    try:
        from statsmodels.discrete.discrete_model import NegativeBinomial
        print("NegativeBinomial: OK")
    except Exception as exc:
        print("NegativeBinomial: MISSING")
        print(exc)

    try:
        import scipy
        print("scipy: OK")
        print("scipy version:", scipy.__version__)
    except Exception as exc:
        print("scipy: MISSING")
        print(exc)


if __name__ == "__main__":
    main()