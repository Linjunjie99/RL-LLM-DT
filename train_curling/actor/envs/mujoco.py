from typing import Any, Tuple

import gym

from core.env import Env


class MujocoEnv(Env):
    def __init__(self, gym_env_id, *args, **kwargs):
        super(MujocoEnv, self).__init__(*args, **kwargs)
        self.env = gym.make(gym_env_id)

    def step(self, action: Any, *args, **kwargs) -> Tuple[Any, Any, Any, Any]:
        return self.env.step(action)

    def reset(self, *args, **kwargs) -> Any:
        return self.env.reset()

    def get_action_space(self) -> Any:
        return self.env.action_space.shape[0]

    def get_observation_space(self) -> Any:
        return self.env.observation_space.shape[0]

    def calc_reward(self, *args, **kwargs) -> Any:
        raise NotImplemented

    def render(self, *args, **kwargs) -> None:
        self.env.render()
