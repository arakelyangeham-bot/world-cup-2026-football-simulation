from dataclasses import dataclass

from shared.competition_registry import get_competition


@dataclass(frozen=True)
class HistoricalMatchDataset:
    competition_key: str
    year: int
    source: str
    filename: str

    @property
    def dataset_id(self) -> str:
        return f"{self.competition_key}_{self.year}"

    @property
    def competition(self):
        return get_competition(self.competition_key)


HISTORICAL_MATCH_DATASETS: list[HistoricalMatchDataset] = [
    HistoricalMatchDataset(
        competition_key="euro",
        year=2012,
        source="sofascore",
        filename="euro_2012_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="euro",
        year=2016,
        source="sofascore",
        filename="euro_2016_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="euro",
        year=2020,
        source="sofascore",
        filename="euro_2020_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="euro",
        year=2024,
        source="sofascore",
        filename="euro_2024_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="world_cup",
        year=2010,
        source="sofascore",
        filename="wc_2010_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="world_cup",
        year=2014,
        source="sofascore",
        filename="wc_2014_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="world_cup",
        year=2018,
        source="sofascore",
        filename="wc_2018_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="world_cup",
        year=2022,
        source="sofascore",
        filename="wc_2022_match_results.csv",
    ),
        HistoricalMatchDataset(
        competition_key="copa_america",
        year=2011,
        source="sofascore",
        filename="copa_america_2011_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="copa_america",
        year=2015,
        source="sofascore",
        filename="copa_america_2015_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="copa_america",
        year=2016,
        source="sofascore",
        filename="copa_america_2016_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="copa_america",
        year=2019,
        source="sofascore",
        filename="copa_america_2019_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="copa_america",
        year=2021,
        source="sofascore",
        filename="copa_america_2021_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="copa_america",
        year=2024,
        source="sofascore",
        filename="copa_america_2024_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2010,
        source="sofascore",
        filename="afcon_2010_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2012,
        source="sofascore",
        filename="afcon_2012_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2013,
        source="sofascore",
        filename="afcon_2013_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2015,
        source="sofascore",
        filename="afcon_2015_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2017,
        source="sofascore",
        filename="afcon_2017_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2019,
        source="sofascore",
        filename="afcon_2019_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2021,
        source="sofascore",
        filename="afcon_2021_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2023,
        source="sofascore",
        filename="afcon_2023_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="afcon",
        year=2025,
        source="sofascore",
        filename="afcon_2025_match_results.csv",
    ),
    HistoricalMatchDataset(
        competition_key="asian_cup",
        year=2011,
        source="sofascore",
        filename="asian_cup_2011_match_results.csv"
    ),
    HistoricalMatchDataset(
        competition_key="asian_cup",
        year=2015,
        source="sofascore",
        filename="asian_cup_2015_match_results.csv"
    ),
    HistoricalMatchDataset(
        competition_key="asian_cup",
        year=2019,
        source="sofascore",
        filename="asian_cup_2019_match_results.csv"
    ),
    HistoricalMatchDataset(
        competition_key="asian_cup",
        year=2023,
        source="sofascore",
        filename="asian_cup_2023_match_results.csv"
    ),

]

def get_available_historical_match_datasets(
    competition_key: str | None = None,
) -> list[HistoricalMatchDataset]:
    datasets = HISTORICAL_MATCH_DATASETS

    if competition_key is not None:
        datasets = [
            dataset
            for dataset in datasets
            if dataset.competition_key == competition_key
        ]

    return list(datasets)