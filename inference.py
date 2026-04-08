import os
from openai import OpenAI

from env.environment import DataCleaningEnv
from env.models import Action
from env.tasks import easy_task
from env.graders import compute_score


# ===============================
# Environment Variables
# ===============================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


# OpenAI Client (required by hackathon)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


# ===============================
# Simple Rule-Based Policy
# ===============================
def simple_policy(state):
    if state.duplicate_rows > 0:
        return "remove_duplicates"
    elif state.missing_values > 0:
        return "fill_missing"
    elif state.outliers > 0:
        return "remove_outliers"
    else:
        return "stop"


# ===============================
# Main Inference Loop
# ===============================
def run():
    env = DataCleaningEnv()

    # Initialize task
    env._state = easy_task()

    print(f"[START] task=easy env=autoprep model={MODEL_NAME}")

    rewards = []
    step_num = 0
    done = False

    try:
        while not done:
            step_num += 1

            # Get current state
            current_state = env.state()

            # Decide action
            action_str = simple_policy(current_state)
            action = Action(action_type=action_str)

            # Take step
            state, reward, done, _ = env.step(action)

            reward_val = round(reward.value, 2)
            rewards.append(f"{reward_val:.2f}")

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward_val:.2f} done={str(done).lower()} error=null"
            )

        # Compute final score
        score = compute_score(state)

        print(
            f"[END] success={str(score == 1.0).lower()} "
            f"steps={step_num} score={score:.2f} rewards={','.join(rewards)}"
        )

    except Exception:
        print(
            f"[END] success=false steps={step_num} score=0.00 "
            f"rewards={','.join(rewards)}"
        )


# ===============================
# Run Entry
# ===============================
if __name__ == "__main__":
    run()