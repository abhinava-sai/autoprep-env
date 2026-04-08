from env.models import Observation


def easy_task():
    return Observation(
        missing_values=20,
        duplicate_rows=10,
        outliers=5,
        step_count=0
    )


def medium_task():
    return Observation(
        missing_values=50,
        duplicate_rows=20,
        outliers=10,
        step_count=0
    )


def hard_task():
    return Observation(
        missing_values=100,
        duplicate_rows=50,
        outliers=30,
        step_count=0
    )