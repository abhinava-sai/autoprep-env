from env.models import Observation


def compute_score(state: Observation) -> float:
    total_issues = (
        state.missing_values +
        state.duplicate_rows +
        state.outliers
    )

    # Maximum possible issues (from hard task)
    max_issues = 100 + 50 + 30  # = 180

    score = 1 - (total_issues / max_issues)

    # Clamp between 0 and 1
    score = max(0.0, min(1.0, score))

    return round(score, 2)