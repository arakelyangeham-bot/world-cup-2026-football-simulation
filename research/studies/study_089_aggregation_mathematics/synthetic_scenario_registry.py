# synthetic_scenario_registry.py

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Iterable, Sequence


ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE = 1e-12

DEFAULT_RANDOM_SEED = 42


VALID_SCENARIO_FAMILIES = {
    "axiom",
    "stability",
    "elite_responsiveness",
    "distribution_shape",
    "depth",
    "rank_boundary",
    "scale_consistency",
    "structural_role",
    "regression",
}


VALID_EXPECTED_DIRECTIONS = {
    "equal",
    "increase",
    "decrease",
    "non_decrease",
    "non_increase",
    "descriptive",
}


@dataclass(frozen=True)
class SyntheticPopulation:
    """
    Immutable one-dimensional synthetic player population.

    The population is intentionally independent of any aggregation
    function. It describes only a controlled football world.
    """

    population_id: str
    values: tuple[float, ...]
    description: str

    def validate(self) -> None:
        if not self.population_id.strip():
            raise ValueError(
                "Synthetic population ID must not be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "Synthetic population description must not be empty."
            )

        if not self.values:
            raise ValueError(
                f"Synthetic population {self.population_id!r} "
                "must contain at least one value."
            )

        for index, value in enumerate(self.values):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    "Synthetic population values must be numeric. "
                    f"Population={self.population_id!r}, "
                    f"index={index}, value={value!r}."
                )

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise ValueError(
                    "Synthetic population values must be finite. "
                    f"Population={self.population_id!r}, "
                    f"index={index}, value={value!r}."
                )

            if not 0.0 <= numeric_value <= 1.0:
                raise ValueError(
                    "Synthetic population values must lie within "
                    "[0, 1]. "
                    f"Population={self.population_id!r}, "
                    f"index={index}, value={value!r}."
                )

    def to_record(self) -> dict[str, object]:
        self.validate()

        return {
            "population_id": self.population_id,
            "description": self.description,
            "player_count": len(self.values),
            "values": ", ".join(
                f"{value:.12f}"
                for value in self.values
            ),
        }


@dataclass(frozen=True)
class SyntheticScenario:
    """
    Immutable Study 089B synthetic benchmark scenario.

    A scenario contains only:

    - controlled player populations;
    - the transformation or comparison being represented;
    - the expected directional behavior;
    - the mathematical or football property under examination.

    It does not know which aggregation functions will be applied.
    """

    scenario_id: str
    scenario_family: str
    name: str
    description: str

    population_a: SyntheticPopulation
    population_b: SyntheticPopulation | None

    evaluated_property: str
    expected_direction: str

    comparison_label_a: str = "baseline"
    comparison_label_b: str = "modified"

    binary_pass_expected: bool = False
    notes: str = ""

    def validate(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError(
                "Scenario ID must not be empty."
            )

        if self.scenario_family not in VALID_SCENARIO_FAMILIES:
            raise ValueError(
                "Unknown scenario family "
                f"{self.scenario_family!r} for "
                f"{self.scenario_id!r}."
            )

        if not self.name.strip():
            raise ValueError(
                f"Scenario {self.scenario_id!r} "
                "must have a name."
            )

        if not self.description.strip():
            raise ValueError(
                f"Scenario {self.scenario_id!r} "
                "must have a description."
            )

        if not self.evaluated_property.strip():
            raise ValueError(
                f"Scenario {self.scenario_id!r} "
                "must identify an evaluated property."
            )

        if (
            self.expected_direction
            not in VALID_EXPECTED_DIRECTIONS
        ):
            raise ValueError(
                "Unknown expected direction "
                f"{self.expected_direction!r} for "
                f"{self.scenario_id!r}."
            )

        self.population_a.validate()

        if self.population_b is not None:
            self.population_b.validate()

        if (
            self.expected_direction
            != "descriptive"
            and self.population_b is None
        ):
            raise ValueError(
                f"Scenario {self.scenario_id!r} expects "
                f"direction {self.expected_direction!r} but "
                "does not define population_b."
            )

        if not self.comparison_label_a.strip():
            raise ValueError(
                f"Scenario {self.scenario_id!r} has an empty "
                "population-A label."
            )

        if (
            self.population_b is not None
            and not self.comparison_label_b.strip()
        ):
            raise ValueError(
                f"Scenario {self.scenario_id!r} has an empty "
                "population-B label."
            )

    def to_record(self) -> dict[str, object]:
        self.validate()

        return {
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "name": self.name,
            "description": self.description,
            "population_a_id": self.population_a.population_id,
            "population_b_id": (
                self.population_b.population_id
                if self.population_b is not None
                else None
            ),
            "comparison_label_a": self.comparison_label_a,
            "comparison_label_b": (
                self.comparison_label_b
                if self.population_b is not None
                else None
            ),
            "evaluated_property": self.evaluated_property,
            "expected_direction": self.expected_direction,
            "binary_pass_expected": self.binary_pass_expected,
            "notes": self.notes,
        }


def _population(
    population_id: str,
    values: Sequence[float],
    description: str,
) -> SyntheticPopulation:
    population = SyntheticPopulation(
        population_id=population_id,
        values=tuple(
            float(value)
            for value in values
        ),
        description=description,
    )

    population.validate()

    return population


def uniform_population(
    *,
    population_id: str,
    value: float,
    size: int,
    description: str,
) -> SyntheticPopulation:
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(
            "Synthetic population size must be an integer."
        )

    if size <= 0:
        raise ValueError(
            "Synthetic population size must be greater than zero."
        )

    return _population(
        population_id,
        [float(value)] * size,
        description,
    )


def remove_value_once(
    values: Sequence[float],
    *,
    value_to_remove: float,
) -> tuple[float, ...]:
    output = list(
        float(value)
        for value in values
    )

    target = float(value_to_remove)

    for index, value in enumerate(output):
        if math.isclose(
            value,
            target,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        ):
            del output[index]
            return tuple(output)

    raise ValueError(
        f"Value {target!r} was not found in the population."
    )


def append_values(
    values: Sequence[float],
    additions: Iterable[float],
) -> tuple[float, ...]:
    return tuple(
        [
            *(
                float(value)
                for value in values
            ),
            *(
                float(value)
                for value in additions
            ),
        ]
    )


def shift_values(
    values: Sequence[float],
    *,
    amount: float,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[float, ...]:
    if not math.isfinite(float(amount)):
        raise ValueError(
            "Shift amount must be finite."
        )

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must not exceed upper_bound."
        )

    return tuple(
        min(
            upper_bound,
            max(
                lower_bound,
                float(value) + float(amount),
            ),
        )
        for value in values
    )


def multiply_values(
    values: Sequence[float],
    *,
    factor: float,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[float, ...]:
    if not math.isfinite(float(factor)):
        raise ValueError(
            "Multiplication factor must be finite."
        )

    if factor < 0.0:
        raise ValueError(
            "Multiplication factor must not be negative."
        )

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must not exceed upper_bound."
        )

    return tuple(
        min(
            upper_bound,
            max(
                lower_bound,
                float(value) * float(factor),
            ),
        )
        for value in values
    )


def build_scenario_registry() -> tuple[SyntheticScenario, ...]:
    scenarios: list[SyntheticScenario] = []

    def register(
        scenario: SyntheticScenario,
    ) -> None:
        scenario.validate()
        scenarios.append(scenario)

    # -----------------------------------------------------------------
    # AXIOM SCENARIOS
    # -----------------------------------------------------------------

    axiom_reference = _population(
        "AX-001-reference",
        [
            0.91,
            0.88,
            0.84,
            0.80,
            0.76,
            0.71,
            0.66,
        ],
        "Canonical population for deterministic repeated evaluation.",
    )

    register(
        SyntheticScenario(
            scenario_id="AX-001",
            scenario_family="axiom",
            name="Determinism",
            description=(
                "Repeated evaluation of an identical population "
                "must produce identical output."
            ),
            population_a=axiom_reference,
            population_b=axiom_reference,
            evaluated_property="determinism",
            expected_direction="equal",
            comparison_label_a="evaluation_1",
            comparison_label_b="evaluation_2",
            binary_pass_expected=True,
        )
    )

    register(
        SyntheticScenario(
            scenario_id="AX-002",
            scenario_family="axiom",
            name="Permutation invariance",
            description=(
                "Changing player ordering must not change the "
                "aggregated representation."
            ),
            population_a=_population(
                "AX-002-canonical",
                axiom_reference.values,
                "Canonical ordering.",
            ),
            population_b=_population(
                "AX-002-permuted",
                [
                    0.76,
                    0.91,
                    0.66,
                    0.84,
                    0.71,
                    0.88,
                    0.80,
                ],
                "Fixed deterministic permutation.",
            ),
            evaluated_property="permutation_invariance",
            expected_direction="equal",
            binary_pass_expected=True,
        )
    )

    monotonic_baseline = (
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
    )

    for rank_index in range(len(monotonic_baseline)):
        improved = list(monotonic_baseline)
        improved[rank_index] += 0.01

        register(
            SyntheticScenario(
                scenario_id=f"AX-003-R{rank_index + 1}",
                scenario_family="axiom",
                name=(
                    "Monotonic improvement "
                    f"at rank {rank_index + 1}"
                ),
                description=(
                    "Increasing one player projection must not "
                    "decrease team strength."
                ),
                population_a=_population(
                    f"AX-003-R{rank_index + 1}-baseline",
                    monotonic_baseline,
                    "Monotonicity baseline population.",
                ),
                population_b=_population(
                    f"AX-003-R{rank_index + 1}-improved",
                    improved,
                    (
                        "Population with one player improved "
                        "by 0.01."
                    ),
                ),
                evaluated_property="monotonicity",
                expected_direction="non_decrease",
                binary_pass_expected=True,
            )
        )

    continuity_baseline = (
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
    )

    continuity_deltas = (
        -0.01,
        -0.0001,
        -0.000001,
        0.000001,
        0.0001,
        0.01,
    )

    continuity_rank_indices = (
        0,
        4,
        5,
    )

    continuity_counter = 1

    for rank_index in continuity_rank_indices:
        for delta in continuity_deltas:
            modified = list(continuity_baseline)
            modified[rank_index] += delta

            register(
                SyntheticScenario(
                    scenario_id=(
                        f"AX-004-{continuity_counter:02d}"
                    ),
                    scenario_family="axiom",
                    name=(
                        "Continuity perturbation "
                        f"rank {rank_index + 1}, "
                        f"delta {delta:+.6f}"
                    ),
                    description=(
                        "Measure local output sensitivity to a "
                        "small player-value perturbation."
                    ),
                    population_a=_population(
                        (
                            f"AX-004-{continuity_counter:02d}"
                            "-baseline"
                        ),
                        continuity_baseline,
                        "Continuity baseline population.",
                    ),
                    population_b=_population(
                        (
                            f"AX-004-{continuity_counter:02d}"
                            "-modified"
                        ),
                        modified,
                        "Continuity-perturbed population.",
                    ),
                    evaluated_property="continuity",
                    expected_direction=(
                        "non_decrease"
                        if delta > 0.0
                        else "non_increase"
                    ),
                    binary_pass_expected=True,
                    notes=(
                        f"input_delta={delta}; "
                        f"zero_based_rank_index={rank_index}"
                    ),
                )
            )

            continuity_counter += 1

    for identity_index, identity_value in enumerate(
        (
            0.00,
            0.25,
            0.50,
            0.75,
            1.00,
        ),
        start=1,
    ):
        population = uniform_population(
            population_id=(
                f"AX-005-{identity_index:02d}-uniform"
            ),
            value=identity_value,
            size=5,
            description=(
                "Uniform population for identity validation."
            ),
        )

        register(
            SyntheticScenario(
                scenario_id=(
                    f"AX-005-{identity_index:02d}"
                ),
                scenario_family="axiom",
                name=(
                    f"Identity at {identity_value:.2f}"
                ),
                description=(
                    "A mean-like aggregator should return the "
                    "common player value."
                ),
                population_a=population,
                population_b=population,
                evaluated_property="identity",
                expected_direction="equal",
                binary_pass_expected=True,
                notes=(
                    f"expected_scalar_value={identity_value}"
                ),
            )
        )

    boundedness_populations = (
        (
            "all_zero",
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "all_one",
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        (
            "mixed_extremes",
            [1.0, 0.75, 0.50, 0.25, 0.0],
        ),
        (
            "fixed_irregular",
            [0.93, 0.81, 0.64, 0.42, 0.17],
        ),
    )

    for boundedness_index, (
        label,
        values,
    ) in enumerate(
        boundedness_populations,
        start=1,
    ):
        population = _population(
            (
                f"AX-006-{boundedness_index:02d}"
                f"-{label}"
            ),
            values,
            "Population used for boundedness validation.",
        )

        register(
            SyntheticScenario(
                scenario_id=(
                    f"AX-006-{boundedness_index:02d}"
                ),
                scenario_family="axiom",
                name=f"Boundedness: {label}",
                description=(
                    "Mean-like scalar output must remain within "
                    "the player-value range."
                ),
                population_a=population,
                population_b=population,
                evaluated_property="boundedness",
                expected_direction="descriptive",
                binary_pass_expected=True,
            )
        )

    # -----------------------------------------------------------------
    # STABILITY SCENARIOS
    # -----------------------------------------------------------------

    register(
        SyntheticScenario(
            scenario_id="ST-001",
            scenario_family="stability",
            name="Weakest primary contributor downgrade",
            description=(
                "Downgrade the weakest member of the primary "
                "five by 0.01."
            ),
            population_a=_population(
                "ST-001-baseline",
                [0.92, 0.88, 0.84, 0.80, 0.76],
                "Stable primary unit.",
            ),
            population_b=_population(
                "ST-001-modified",
                [0.92, 0.88, 0.84, 0.80, 0.75],
                "Weakest primary contributor downgraded.",
            ),
            evaluated_property="ordinary_replacement_stability",
            expected_direction="decrease",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="ST-002",
            scenario_family="stability",
            name="Elite contributor downgrade",
            description=(
                "Downgrade the strongest member of the primary "
                "five by 0.01."
            ),
            population_a=_population(
                "ST-002-baseline",
                [0.92, 0.88, 0.84, 0.80, 0.76],
                "Stable primary unit.",
            ),
            population_b=_population(
                "ST-002-modified",
                [0.91, 0.88, 0.84, 0.80, 0.76],
                "Strongest primary contributor downgraded.",
            ),
            evaluated_property="elite_local_sensitivity",
            expected_direction="decrease",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="ST-003",
            scenario_family="stability",
            name="Near-identical fifth/sixth swap",
            description=(
                "Exchange two nearly identical players around "
                "the top-five threshold."
            ),
            population_a=_population(
                "ST-003-baseline",
                [
                    0.90,
                    0.85,
                    0.80,
                    0.75,
                    0.7001,
                    0.7000,
                ],
                "Threshold baseline population.",
            ),
            population_b=_population(
                "ST-003-modified",
                [
                    0.90,
                    0.85,
                    0.80,
                    0.75,
                    0.7000,
                    0.7001,
                ],
                "Equivalent values in swapped input order.",
            ),
            evaluated_property="threshold_stability",
            expected_direction="equal",
        )
    )

    # -----------------------------------------------------------------
    # ELITE RESPONSIVENESS SCENARIOS
    # -----------------------------------------------------------------

    elite_reference = (
        0.98,
        0.86,
        0.84,
        0.82,
        0.80,
        0.76,
    )

    register(
        SyntheticScenario(
            scenario_id="ER-001",
            scenario_family="elite_responsiveness",
            name="Elite player removal",
            description=(
                "Remove the strongest player from a six-player "
                "primary candidate population."
            ),
            population_a=_population(
                "ER-001-baseline",
                elite_reference,
                "Primary unit containing one elite player.",
            ),
            population_b=_population(
                "ER-001-modified",
                remove_value_once(
                    elite_reference,
                    value_to_remove=0.98,
                ),
                "Population after elite-player removal.",
            ),
            evaluated_property="elite_removal_responsiveness",
            expected_direction="decrease",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="ER-002",
            scenario_family="elite_responsiveness",
            name="Ordinary starter removal",
            description=(
                "Remove an ordinary primary contributor while "
                "retaining the elite player."
            ),
            population_a=_population(
                "ER-002-baseline",
                elite_reference,
                "Primary unit containing one elite player.",
            ),
            population_b=_population(
                "ER-002-modified",
                remove_value_once(
                    elite_reference,
                    value_to_remove=0.80,
                ),
                "Population after ordinary-player removal.",
            ),
            evaluated_property="ordinary_removal_responsiveness",
            expected_direction="decrease",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="ER-003",
            scenario_family="elite_responsiveness",
            name="Superstar addition",
            description=(
                "Add an elite player to a balanced primary "
                "population."
            ),
            population_a=_population(
                "ER-003-baseline",
                [0.84, 0.83, 0.82, 0.81, 0.80],
                "Balanced primary population.",
            ),
            population_b=_population(
                "ER-003-modified",
                [0.99, 0.84, 0.83, 0.82, 0.81, 0.80],
                "Balanced population with a superstar added.",
            ),
            evaluated_property="elite_addition_responsiveness",
            expected_direction="increase",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="ER-004",
            scenario_family="elite_responsiveness",
            name="Weak fringe addition",
            description=(
                "Add one weak fringe player below both primary "
                "and replacement groups."
            ),
            population_a=_population(
                "ER-004-baseline",
                [
                    0.90,
                    0.86,
                    0.82,
                    0.78,
                    0.74,
                    0.70,
                    0.66,
                    0.62,
                    0.58,
                    0.54,
                ],
                "Ten-player primary and replacement population.",
            ),
            population_b=_population(
                "ER-004-modified",
                [
                    0.90,
                    0.86,
                    0.82,
                    0.78,
                    0.74,
                    0.70,
                    0.66,
                    0.62,
                    0.58,
                    0.54,
                    0.20,
                ],
                "Population with one weak fringe addition.",
            ),
            evaluated_property="fringe_sensitivity",
            expected_direction="descriptive",
        )
    )

    # -----------------------------------------------------------------
    # DISTRIBUTION-SHAPE SCENARIOS
    # -----------------------------------------------------------------

    register(
        SyntheticScenario(
            scenario_id="DS-001",
            scenario_family="distribution_shape",
            name="Top-heavy versus balanced",
            description=(
                "Compare equal-mean primary populations with "
                "different quality distributions."
            ),
            population_a=_population(
                "DS-001-top-heavy",
                [0.95, 0.90, 0.85, 0.80, 0.75],
                "Top-heavy primary population.",
            ),
            population_b=uniform_population(
                population_id="DS-001-balanced",
                value=0.85,
                size=5,
                description="Balanced equal-mean population.",
            ),
            evaluated_property="distribution_separation",
            expected_direction="descriptive",
            comparison_label_a="top_heavy",
            comparison_label_b="balanced",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="DS-002",
            scenario_family="distribution_shape",
            name="Extreme superstar versus balanced",
            description=(
                "Compare equal-mean populations where one is "
                "dominated by an extreme superstar."
            ),
            population_a=_population(
                "DS-002-superstar",
                [0.99, 0.70, 0.70, 0.70, 0.70],
                "Extreme superstar population.",
            ),
            population_b=uniform_population(
                population_id="DS-002-balanced",
                value=0.758,
                size=5,
                description="Balanced equal-mean population.",
            ),
            evaluated_property="superstar_concentration",
            expected_direction="descriptive",
            comparison_label_a="superstar",
            comparison_label_b="balanced",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="DS-003",
            scenario_family="distribution_shape",
            name="Elite core versus balanced core",
            description=(
                "Compare a varied elite core with a uniform "
                "core having the same arithmetic mean."
            ),
            population_a=_population(
                "DS-003-elite-core",
                [0.92, 0.90, 0.88, 0.86, 0.84],
                "Varied elite primary core.",
            ),
            population_b=uniform_population(
                population_id="DS-003-balanced-core",
                value=0.88,
                size=5,
                description="Balanced equal-mean elite core.",
            ),
            evaluated_property="primary_distribution_shape",
            expected_direction="descriptive",
            comparison_label_a="elite_core",
            comparison_label_b="balanced_core",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="DS-004",
            scenario_family="distribution_shape",
            name="Strong core with weak replacements",
            description=(
                "Compare a strong first unit with poor depth "
                "against a more balanced ten-player population."
            ),
            population_a=_population(
                "DS-004-strong-core",
                [
                    0.92,
                    0.90,
                    0.88,
                    0.86,
                    0.84,
                    0.60,
                    0.58,
                    0.56,
                    0.54,
                    0.52,
                ],
                "Strong primary core with weak replacements.",
            ),
            population_b=_population(
                "DS-004-balanced-depth",
                [
                    0.84,
                    0.83,
                    0.82,
                    0.81,
                    0.80,
                    0.79,
                    0.78,
                    0.77,
                    0.76,
                    0.75,
                ],
                "Balanced primary and replacement quality.",
            ),
            evaluated_property="strength_depth_tradeoff",
            expected_direction="descriptive",
            comparison_label_a="strong_core",
            comparison_label_b="balanced_depth",
        )
    )

    # -----------------------------------------------------------------
    # DEPTH SCENARIOS
    # -----------------------------------------------------------------

    depth_reference = (
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
    )

    register(
        SyntheticScenario(
            scenario_id="DP-001",
            scenario_family="depth",
            name="Roster-size dilution",
            description=(
                "Append weak fringe players below the useful "
                "replacement group."
            ),
            population_a=_population(
                "DP-001-baseline",
                depth_reference,
                "Ten-player primary and replacement population.",
            ),
            population_b=_population(
                "DP-001-expanded",
                append_values(
                    depth_reference,
                    [0.30, 0.25, 0.20, 0.15, 0.10],
                ),
                "Population expanded with weak fringe players.",
            ),
            evaluated_property="roster_size_dilution",
            expected_direction="descriptive",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="DP-002",
            scenario_family="depth",
            name="Replacement-unit improvement",
            description=(
                "Improve only ranks six through ten while "
                "holding the primary five fixed."
            ),
            population_a=_population(
                "DP-002-baseline",
                [
                    0.90,
                    0.88,
                    0.86,
                    0.84,
                    0.82,
                    0.65,
                    0.63,
                    0.61,
                    0.59,
                    0.57,
                ],
                "Strong primary unit with weak replacements.",
            ),
            population_b=_population(
                "DP-002-improved",
                [
                    0.90,
                    0.88,
                    0.86,
                    0.84,
                    0.82,
                    0.75,
                    0.73,
                    0.71,
                    0.69,
                    0.67,
                ],
                "Same primary unit with improved replacements.",
            ),
            evaluated_property="replacement_quality",
            expected_direction="descriptive",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="DP-003",
            scenario_family="depth",
            name="Uniform full-population improvement",
            description=(
                "Improve every player by 0.02 while preserving "
                "the shape of the quality distribution."
            ),
            population_a=_population(
                "DP-003-baseline",
                depth_reference,
                "Baseline primary and replacement population.",
            ),
            population_b=_population(
                "DP-003-improved",
                shift_values(
                    depth_reference,
                    amount=0.02,
                ),
                "Uniformly improved population.",
            ),
            evaluated_property="uniform_quality_shift",
            expected_direction="descriptive",
        )
    )

    # -----------------------------------------------------------------
    # RANK-BOUNDARY SCENARIOS
    # -----------------------------------------------------------------

    register(
        SyntheticScenario(
            scenario_id="RB-001",
            scenario_family="rank_boundary",
            name="Single fifth/sixth crossing",
            description=(
                "Move the sixth-ranked value slightly above the "
                "fifth-ranked threshold."
            ),
            population_a=_population(
                "RB-001-baseline",
                [
                    0.90,
                    0.85,
                    0.80,
                    0.75,
                    0.7000,
                    0.6999,
                ],
                "Population immediately below a top-five crossing.",
            ),
            population_b=_population(
                "RB-001-modified",
                [
                    0.90,
                    0.85,
                    0.80,
                    0.75,
                    0.7000,
                    0.7001,
                ],
                "Population immediately above a top-five crossing.",
            ),
            evaluated_property="single_threshold_crossing",
            expected_direction="non_decrease",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="RB-002",
            scenario_family="rank_boundary",
            name="Three-way tied boundary",
            description=(
                "Audit deterministic behavior when multiple "
                "players share the threshold value."
            ),
            population_a=_population(
                "RB-002-canonical",
                [
                    0.90,
                    0.85,
                    0.80,
                    0.75,
                    0.70,
                    0.70,
                    0.70,
                ],
                "Canonical tied-threshold population.",
            ),
            population_b=_population(
                "RB-002-permuted",
                [
                    0.70,
                    0.90,
                    0.70,
                    0.80,
                    0.75,
                    0.70,
                    0.85,
                ],
                "Permutation of the tied-threshold population.",
            ),
            evaluated_property="tie_handling",
            expected_direction="equal",
            binary_pass_expected=True,
        )
    )

    register(
        SyntheticScenario(
            scenario_id="RB-003",
            scenario_family="rank_boundary",
            name="Cluster around top-five boundary",
            description=(
                "Perturb a dense cluster of values surrounding "
                "the top-five threshold."
            ),
            population_a=_population(
                "RB-003-baseline",
                [
                    0.90,
                    0.85,
                    0.80,
                    0.75,
                    0.7000,
                    0.6999,
                    0.6998,
                    0.6997,
                    0.6996,
                ],
                "Dense threshold cluster before perturbation.",
            ),
            population_b=_population(
                "RB-003-modified",
                [
                    0.90,
                    0.85,
                    0.80,
                    0.75,
                    0.7002,
                    0.7001,
                    0.7000,
                    0.6999,
                    0.6998,
                ],
                "Dense threshold cluster after small perturbation.",
            ),
            evaluated_property="clustered_threshold_sensitivity",
            expected_direction="non_decrease",
        )
    )

    # -----------------------------------------------------------------
    # SCALE-CONSISTENCY SCENARIOS
    # -----------------------------------------------------------------

    scale_reference = (
        0.80,
        0.78,
        0.76,
        0.74,
        0.72,
    )

    register(
        SyntheticScenario(
            scenario_id="SC-001",
            scenario_family="scale_consistency",
            name="Uniform additive shift",
            description=(
                "Increase every player projection by 0.01."
            ),
            population_a=_population(
                "SC-001-baseline",
                scale_reference,
                "Baseline quality population.",
            ),
            population_b=_population(
                "SC-001-shifted",
                shift_values(
                    scale_reference,
                    amount=0.01,
                ),
                "Population shifted upward by 0.01.",
            ),
            evaluated_property="translation_behavior",
            expected_direction="increase",
            notes="expected_arithmetic_delta=0.01",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="SC-002",
            scenario_family="scale_consistency",
            name="Uniform multiplicative increase",
            description=(
                "Multiply every player projection by 1.05."
            ),
            population_a=_population(
                "SC-002-baseline",
                scale_reference,
                "Baseline quality population.",
            ),
            population_b=_population(
                "SC-002-multiplied",
                multiply_values(
                    scale_reference,
                    factor=1.05,
                ),
                "Population multiplied by 1.05.",
            ),
            evaluated_property="multiplicative_behavior",
            expected_direction="increase",
            notes="multiplication_factor=1.05",
        )
    )

    register(
        SyntheticScenario(
            scenario_id="SC-003",
            scenario_family="scale_consistency",
            name="Equivalent normalized population",
            description=(
                "Retain a normalized control population for "
                "later comparison with externally rescaled "
                "equivalents."
            ),
            population_a=_population(
                "SC-003-normalized-a",
                scale_reference,
                "Normalized control population A.",
            ),
            population_b=_population(
                "SC-003-normalized-b",
                scale_reference,
                "Equivalent normalized control population B.",
            ),
            evaluated_property="normalized_scale_equivalence",
            expected_direction="equal",
            binary_pass_expected=True,
            notes=(
                "The [0,100] representation is handled in the "
                "runner as a scale-audit transformation because "
                "SyntheticPopulation enforces normalized [0,1] "
                "inputs."
            ),
        )
    )

    # Structural-role scenarios are intentionally deferred.
    # They require richer player records and should not block the
    # one-dimensional Study 089B benchmark.

    validate_scenario_registry(
        tuple(scenarios)
    )

    return tuple(scenarios)


def validate_scenario_registry(
    scenarios: tuple[SyntheticScenario, ...],
) -> None:
    if not scenarios:
        raise ValueError(
            "Synthetic scenario registry must not be empty."
        )

    scenario_ids = [
        scenario.scenario_id
        for scenario in scenarios
    ]

    duplicate_ids = sorted(
        {
            scenario_id
            for scenario_id in scenario_ids
            if scenario_ids.count(scenario_id) > 1
        }
    )

    if duplicate_ids:
        raise ValueError(
            "Synthetic scenario registry contains duplicate "
            f"scenario IDs: {duplicate_ids}"
        )

    population_registry: dict[
        str,
        SyntheticPopulation,
    ] = {}

    for scenario in scenarios:
        scenario.validate()

        for population in (
            scenario.population_a,
            scenario.population_b,
        ):
            if population is None:
                continue

            existing = population_registry.get(
                population.population_id
            )

            if existing is None:
                population_registry[
                    population.population_id
                ] = population
                continue

            if existing != population:
                raise ValueError(
                    "Synthetic population ID "
                    f"{population.population_id!r} "
                    "is associated with multiple different "
                    "population definitions."
                )


def scenario_registry_records(
    scenarios: tuple[
        SyntheticScenario,
        ...,
    ] | None = None,
) -> list[dict[str, object]]:
    selected = (
        scenarios
        if scenarios is not None
        else build_scenario_registry()
    )

    validate_scenario_registry(
        selected
    )

    return [
        scenario.to_record()
        for scenario in selected
    ]


def population_registry_records(
    scenarios: tuple[
        SyntheticScenario,
        ...,
    ] | None = None,
) -> list[dict[str, object]]:
    selected = (
        scenarios
        if scenarios is not None
        else build_scenario_registry()
    )

    validate_scenario_registry(
        selected
    )

    records: list[dict[str, object]] = []

    for scenario in selected:
        for label, population in (
            (
                scenario.comparison_label_a,
                scenario.population_a,
            ),
            (
                scenario.comparison_label_b,
                scenario.population_b,
            ),
        ):
            if population is None:
                continue

            for player_rank, value in enumerate(
                population.values,
                start=1,
            ):
                records.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "scenario_family":
                            scenario.scenario_family,
                        "population_label": label,
                        "population_id":
                            population.population_id,
                        "population_description":
                            population.description,
                        "source_order": player_rank,
                        "player_value": value,
                    }
                )

    return records