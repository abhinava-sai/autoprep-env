import os
from openai import OpenAI

from env.environment import DataCleaningEnv
from env.models import Action
from env.tasks import easy_task
from env.graders import compute_score


# ===============================
# Environment Variables (STRICT)
# ===============================
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.environ["HF_TOKEN"]   # ← IMPORTANT FIX


# ===============================
# OpenAI Client
# ===============================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


# ===============================
# Valid Actions
# ===============================
VALID_ACTIONS = [
    "remove_duplicates",
    "fill_missing",
    "remove_outliers",
    "stop"
]


# ===============================
# LLM Policy
# ===============================
def llm_policy(state):
    prompt = f"""
You are a data cleaning agent.

Current dataset state:
- duplicate_rows: {state.duplicate_rows}
- missing_values: {state.missing_values}
- outliers: {state.outliers}

Choose EXACTLY ONE action from:
remove_duplicates, fill_missing, remove_outliers, stop

Respond ONLY with the action name.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise data cleaning agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        action = response.choices[0].message.content.strip()

        if action not in VALID_ACTIONS:
            action = "stop"

        return action

    except Exception:
        # fallback (VERY IMPORTANT)
        if state.duplicate_rows > 0:
            return "remove_duplicates"
        elif state.missing_values > 0:
            return "fill_missing"
        elif state.outliers > 0:
            return "remove_outliers"
        else:
            return "stop"


# ===============================
# Main Loop
# ===============================
def run():
    env = DataCleaningEnv()
    env._state = easy_task()

    print(f"[START] task=easy env=autoprep model={MODEL_NAME}")

    rewards = []
    step_num = 0
    done = False

    try:
        while not done:
            step_num += 1

            current_state = env.state()

            # MUST call LLM
            action_str = llm_policy(current_state)
            action = Action(action_type=action_str)

            state, reward, done, info = env.step(action)

            reward_val = float(reward.value)
            rewards.append(f"{reward_val:.2f}")

            error_msg = "null"
            if info and info.get("last_action_error"):
                error_msg = info["last_action_error"]

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward_val:.2f} done={str(done).lower()} error={error_msg}"
            )

        score = float(compute_score(state))

        print(
            f"[END] success={str(score == 1.0).lower()} "
            f"steps={step_num} score={score:.2f} rewards={','.join(rewards)}"
        )

    except Exception:
        print(
            f"[END] success=false steps={step_num} score=0.00 "
            f"rewards={','.join(rewards)}"
        )


if __name__ == "__main__":
    run()