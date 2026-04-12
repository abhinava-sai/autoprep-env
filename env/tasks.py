from env.models import Observation


# ===============================
# EASY TASK
# Simple cleaning (no outliers)
# ===============================
def easy_task():
    return Observation(
        missing_values=10,
        duplicate_rows=5,
        outliers=0,
        step_count=0
    )


# ===============================
# MEDIUM TASK
# Mixed cleaning (balanced difficulty)
# ===============================
def medium_task():
    return Observation(
        missing_values=25,
        duplicate_rows=10,
        outliers=15,
        step_count=0
    )


# ===============================
# HARD TASK
# Outlier-heavy (requires prioritization)
# ===============================
def hard_task():
    return Observation(
        missing_values=40,
        duplicate_rows=5,
        outliers=30,
        step_count=0
    )