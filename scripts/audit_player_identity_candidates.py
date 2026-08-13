#audit_player_identity_candidates

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

OUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_player_identity_candidates.csv"
)


def main():
    df = pd.read_csv(
        STATS_FILE,
        dtype={"season_year": str},
        low_memory=False,
    )

    required_columns = [
        "player_id",
        "player",
        "competition",
        "season_year",
        "team",
        "minutesPlayed",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    evidence = (
        df[
            required_columns
        ]
        .dropna(
            subset=[
                "player_id",
                "competition",
                "season_year",
                "team",
                "minutesPlayed",
            ]
        )
        .copy()
    )

    evidence["player_id"] = (
        evidence["player_id"]
        .astype(int)
    )

    evidence["minutesPlayed"] = pd.to_numeric(
        evidence["minutesPlayed"],
        errors="coerce",
    )

    evidence = evidence.dropna(
        subset=["minutesPlayed"]
    )

    grouped = (
        evidence
        .groupby(
            [
                "competition",
                "season_year",
                "team",
                "minutesPlayed",
            ],
            dropna=False,
        )
    )

    candidates = []

    for key, group in grouped:
        unique_ids = (
            group[
                ["player_id", "player"]
            ]
            .drop_duplicates(
                subset=["player_id"]
            )
        )

        if len(unique_ids) < 2:
            continue

        rows = unique_ids.to_dict(
            "records"
        )

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left = rows[i]
                right = rows[j]

                candidates.append({
                    "player_id_a": left["player_id"],
                    "player_a": left["player"],
                    "player_id_b": right["player_id"],
                    "player_b": right["player"],
                    "competition": key[0],
                    "season_year": key[1],
                    "team": key[2],
                    "minutesPlayed": key[3],
                })

    out = pd.DataFrame(candidates)

    if not out.empty:
        out = (
            out
            .sort_values(
                [
                    "team",
                    "season_year",
                    "competition",
                    "minutesPlayed",
                    "player_id_a",
                    "player_id_b",
                ]
            )
            .reset_index(drop=True)
        )

    OUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT_FILE,
        index=False,
    )

    print(
        f"Candidate rows: {len(out):,}"
    )

    if out.empty:
        print(
            "No duplicate-evidence identity "
            "candidates found."
        )
        return

    pair_summary = (
        out
        .groupby(
            [
                "player_id_a",
                "player_a",
                "player_id_b",
                "player_b",
            ]
        )
        .size()
        .reset_index(
            name="matching_evidence_rows"
        )
        .sort_values(
            "matching_evidence_rows",
            ascending=False,
        )
    )

    print()
    print(
        "Top candidate identity pairs:"
    )
    print(
        pair_summary.head(50).to_string(
            index=False
        )
    )

    print()
    print(f"Wrote: {OUT_FILE}")


if __name__ == "__main__":
    main()