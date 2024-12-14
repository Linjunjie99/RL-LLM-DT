import sys
sys.path.append("../..")
from core.env import Env
from .chooseenv import *
# from .obs_interfaces.observation import *
from .common import obs_pre_process

class RunningEnv(Env):
    def __init__(self):
        super(RunningEnv, self).__init__()
        self.env = make("olympics-running")
        self.num_agents = self.env.n_player
        # self.width = self.env_core.view_setting['width']+2*self.env_core.view_setting['edge']
        # self.height = self.env_core.view_setting['height']+2*self.env_core.view_setting['edge']
        self.act_dim = (2)
        self.obs_dim = (25, 25, 4)

    def reset(self):
        obs = self.env.reset(shuffle_map=True)
        # state = obs_pre_process(obs)
        return obs

    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        return next_state, reward, done, _, info


    def get_action_space(self):
        return self.act_dim

    def get_observation_space(self):
        return self.obs_dim

    def render(self) -> None:
        self.render()

    def calc_reward(self, *args, **kwargs):
        raise NotImplemented


# if __name__ == "__main__":
#     env = RunningEnv()
#     state = env.reset()
#     print(f'state is {state}')
