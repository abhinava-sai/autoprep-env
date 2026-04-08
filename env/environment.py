from env.models import Observation, Action, Reward


class DataCleaningEnv:
    def __init__(self):
        self._state = None   # 🔥 renamed to avoid conflict
        self.step_count = 0
        self.max_steps = 10

    def reset(self):
        self._state = Observation(
            missing_values=50,
            duplicate_rows=20,
            outliers=10,
            step_count=0
        )
        self.step_count = 0
        return self._state

    def state(self):   # 🔥 REQUIRED by OpenEnv
        return self._state

    def step(self, action: Action):
        self.step_count += 1
        done = False

        # Previous total issues
        prev_total = (
            self._state.missing_values +
            self._state.duplicate_rows +
            self._state.outliers
        )

        # Apply action
        if action.action_type == "remove_duplicates":
            self._state.duplicate_rows = max(0, self._state.duplicate_rows - 10)

        elif action.action_type == "fill_missing":
            self._state.missing_values = max(0, self._state.missing_values - 10)

        elif action.action_type == "remove_outliers":
            self._state.outliers = max(0, self._state.outliers - 5)

        elif action.action_type == "stop":
            done = True

        else:
            reward = Reward(value=-0.2, reason="invalid_action")
            return self._state, reward, done, {}

        # New total
        new_total = (
            self._state.missing_values +
            self._state.duplicate_rows +
            self._state.outliers
        )

        # Reward = improvement
        reward_value = (prev_total - new_total) / 100.0

        # Step penalty
        reward_value -= 0.05

        # No improvement penalty
        if prev_total == new_total:
            reward_value -= 0.1

        # Success condition
        if (
            self._state.missing_values == 0 and
            self._state.duplicate_rows == 0 and
            self._state.outliers == 0
        ):
            done = True
            reward_value += 1.0

        # Max step limit
        if self.step_count >= self.max_steps:
            done = True

        # Update step count
        self._state.step_count = self.step_count

        reward = Reward(value=round(reward_value, 2))

        return self._state, reward, done, {}