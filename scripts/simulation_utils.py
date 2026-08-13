#simulation_utils.py

def canonical_scoreline(goals_a: int, goals_b: int) -> str:
    high = max(goals_a, goals_b)
    low = min(goals_a, goals_b)
    return f"{high}-{low}"