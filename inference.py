import os
from openai import OpenAI

from env.environment import DataCleaningEnv
from env.models import Action
from env.tasks import easy_task
from env.graders import compute_score


# ===============================
# Environment Variables (FINAL SAFE)
# ===============================
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["HF_TOKEN"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")


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
# LLM Policy (Robust)
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

        action = (response.choices[0].message.content or "").strip()

        if action not in VALID_ACTIONS:
            action = "stop"

        return action

    except Exception:
        # fallback ensures execution continues
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

    rewards = []
    step_num = 0
    done = False
    state = None  # safety

    print(f"[START] task=easy env=autoprep model={MODEL_NAME}")

    try:
        while not done:
            step_num += 1

            current_state = env.state()

            # MUST attempt LLM call
            action_str = llm_policy(current_state)
            action = Action(action_type=action_str)

            try:
                state, reward, done, info = env.step(action)
            except Exception:
                # step failure safety
                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward=0.00 done=true error=step_error"
                )
                break

            # safe reward extraction
            reward_val = 0.0
            try:
                reward_val = float(reward.value)
            except Exception:
                reward_val = 0.0

            rewards.append(f"{reward_val:.2f}")

            # safe error handling
            error_msg = "null"
            try:
                if info and info.get("last_action_error"):
                    error_msg = info["last_action_error"]
            except Exception:
                error_msg = "null"

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward_val:.2f} done={str(done).lower()} error={error_msg}"
            )

        # safe score computation
        score = 0.0
        try:
            if state is not None:
                score = float(compute_score(state))
        except Exception:
            score = 0.0

        print(
            f"[END] success={str(score == 1.0).lower()} "
            f"steps={step_num} score={score:.2f} rewards={','.join(rewards)}"
        )

    except Exception:
        # global safety (MANDATORY END)
        print(
            f"[END] success=false steps={step_num} score=0.00 "
            f"rewards={','.join(rewards)}"
        )


# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()