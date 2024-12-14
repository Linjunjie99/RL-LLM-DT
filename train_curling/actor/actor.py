import os
import re
import sys
import csv
os.environ["KMP_WARNINGS"] = "FALSE" 
from env.chooseenv import make
import time
from argparse import ArgumentParser
from collections import deque
from itertools import count
from multiprocessing import Array, Process
import copy
import random

import numpy as np
import zmq
from pyarrow import serialize

import math
from envs import get_env
from models import get_model
from utils import logger
from utils.cmdline import parse_cmdline_kwargs
from utils.data_trans import (create_experiment_dir, find_new_weights,
                              load_yaml_config, prepare_training_data)


parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='ppo_con', help='The RL algorithm')
parser.add_argument('--env', type=str, default='olympics-running', help='The game environment')
parser.add_argument('--num_steps', type=int, default=10000000, help='The number of total training steps')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server port to subscribe model parameters')
parser.add_argument('--num_replicas', type=int, default=40, help='The number of actors')
parser.add_argument('--max_episodes', type=int, default=100000000000, help='The number of actors')
parser.add_argument('--model', type=str, default='accnn_con', help='Training model')
parser.add_argument('--max_steps_per_update', type=int, default=128,
                    help='The maximum number of steps between each update')
parser.add_argument('--exp_path', type=str, default=None,
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--num_saved_ckpt', type=int, default=10, help='Number of recent checkpoint files to be saved')
parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum length of trajectory')
parser.add_argument('--config', type=str, default=None, help='The YAML configuration file')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU to sample every action')
parser.add_argument('--num_envs', type=int, default=1, help='The number of environment copies')

class Obs_Process():
    def __init__(self, obs):

        self.fix_x = 30.5
        self.fix_y = 14.5
        self.red = 7
        self.grey = 4
        self.obs = obs

        self.min_cor = -60
        self.image = self.obs['obs'][0]


    def findball(self, obs, color):
        result = []
        pic = copy.deepcopy(obs)
        for i in range(0, len(pic)):
            for j in range(0, len(pic)):
                if pic[i, j] != color:
                    pic[i, j] = 0
                else:
                    pic[i, j] = 1
        cor = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],

                        [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
                        [[1, 1, 1], [1, 1, 1], [0, 1, 1]],
                        [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 0], [1, 1, 1], [1, 1, 1]],

                        [[0, 1, 0], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [0, 1, 0]],
                        [[0, 1, 1], [1, 1, 1], [0, 1, 1]],
                        [[1, 1, 0], [1, 1, 1], [1, 1, 0]],

                        [[0, 1, 0], [1, 1, 1], [1, 1, 0]],
                        [[1, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 1, 1], [1, 1, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 1]],

                        [[0, 0, 0], [1, 1, 1], [1, 1, 1]],
                        [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
                        [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],

                        [[0, 0, 0], [1, 1, 1], [0, 1, 1]],
                        [[0, 1, 1], [0, 1, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, 1, 0], [1, 1, 0]],
                        [[0, 1, 1], [1, 1, 1], [0, 0, 0]],

                        [[0, 0, 0], [1, 1, 1], [1, 1, 0]],
                        [[0, 1, 0], [0, 1, 1], [0, 1, 1]],
                        [[1, 1, 0], [1, 1, 0], [0, 1, 0]],
                        [[1, 1, 0], [1, 1, 1], [0, 0, 0]],

                        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                        [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
                        [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                        [[0, 0, 0], [1, 1, 0], [1, 1, 0]]], dtype=np.float64)

        for item in cor:
            for i in range(1, len(pic) - 1):
                for j in range(1, len(pic) - 1):
                    if (item == pic[i - 1:i + 2, j - 1:j + 2]).all():
                        b = np.nonzero(pic[i - 1:i + 2, j - 1:j + 2])
                        pic[i - 1:i + 2, j - 1:j + 2] = 0
                        result.append([np.mean(b[1]+j-1), np.mean(b[0]+i-1)])

        return result

    def obs_pre_process(self,):
        pic = copy.deepcopy(self.obs['obs'][0])
        origin = np.array([14.5, 26.5])
        center = np.array([14.5, 11.5])


        green_ball = self.findball(pic, 1.)
        # print(green_ball)
        purple_ball = self.findball(pic, 5.)
        # print(purple_ball)

        if self.obs['team color'] == 'purple':
            enemy = green_ball
            team = purple_ball
        else:
            enemy = purple_ball
            team = green_ball
        center = (origin - center) / 20
        enemy = [np.array([(item[0] - origin[0]) / 20, (origin[1] - item[1]) / 20]) for item in enemy]
        team = [np.array([(item[0] - origin[0]) / 20, (origin[1] - item[1]) / 20]) for item in team]
        while len(team) < 3:
            team.append(np.array([-3, -3]))
        while len(enemy) < 4:
            enemy.append(np.array([-3, -3]))

        return team, enemy, center

class fix_law():
    def __init__(self) -> None:
        self.team = None
        self.enemy = None
        self.center = None
        self.first_to_move = None
        self.tl = None

        self.mes = [[0.6165146039598741, -1.1177064422806005],
        [0.0, -1.1177064422806005],
        [-0.6165146039598739, -1.1177064422806005],]

        self.pos = 0
        self.angle = 0
        self.r_min = 0.0715#壶半径15，像素1.5，坐标系中0.075
        self.r_max = 0.080

    def set(self, team=None, enemy=None, center=None, first_to_move=None, tl=None):
        if team is not None:
            self.team = []
            for i in range(3):
                if team[0][2 * i] != -3:
                    self.team.append([team[0][2 * i], team[0][2 * i + 1]]) 
        
        if enemy is not None:
            self.enemy = []
            for i in range(4):
                if enemy[0][2 * i] != -3:
                    self.enemy.append([enemy[0][2 * i], enemy[0][2 * i + 1]]) 

        if center is not None:        
            self.center = center[0]
        
        if first_to_move is not None:
            self.first_to_move = first_to_move[0]
        
        if tl is not None:
            self.tl = tl[0]

        self.set_pos()
    
    def set_pos(self,):
        if len(self.enemy) == 0:
            self.pos = 0
            self.angle = 0
        else:
            distan = []
            for item in self.enemy:
                temp = item - self.center
                distan.append(temp[0]**2 + temp[1]**2)
            
            focus_idx = distan.index(min(distan))

            no_shelter = []#不可能遮挡的
            po_shelter = []#可能遮挡的
            po_shelter_dist = []
            for i in range(len(self.mes)):
                min_dist = self.shelter_min_dist(self.mes[i], focus_idx)
                if min_dist > 2 * self.r_max:
                    no_shelter.append(i)
                elif min_dist > 2 * self.r_min:
                    po_shelter.append(i)
                    po_shelter_dist.append(min_dist)
            angles = []
            abs_angles = []
            if len(no_shelter) != 0:
                for i in no_shelter:
                    angel_r = np.arctan((self.enemy[focus_idx][0] - self.mes[i][0])/(self.enemy[focus_idx][1] - self.mes[i][1]))
                    angle = (angel_r/np.pi) * 180
                    angles.append(angle)
                    abs_angles.append(abs(angle))
                
                idx = abs_angles.index(min(abs_angles))
                self.pos = no_shelter[idx] + 1
                self.angle = angles[idx]
            elif len(po_shelter) != 0:
                idx = po_shelter_dist.index(max(po_shelter_dist))
                angel_r = np.arctan((self.enemy[focus_idx][0] - self.mes[po_shelter[idx]][0])/(self.enemy[focus_idx][1] - self.mes[po_shelter[idx]][1]))
                angle = (angel_r/np.pi) * 180
                self.pos = po_shelter[idx] + 1
                self.angle = angle
            else:
                self.pos = 0

                
    def shelter_min_dist(self, me, focus_idx):
        focus_ene = self.enemy[focus_idx]

        A = focus_ene[1] - me[1]
        B = me[0] - focus_ene[0]
        C = focus_ene[0] * me[1] - me[0] * focus_ene[1]
        min_dist = 1e4
        for temp in self.team:
            dist = abs(A * temp[0] + B * temp[1] + C)/math.sqrt(A ** 2 + B ** 2)
            if dist < min_dist:
                min_dist = dist
        
        for temp in self.enemy:
            if temp[0] == focus_ene[0] and temp[1] == focus_ene[1]:
                continue

            dist = abs(A * temp[0] + B * temp[1] + C)/math.sqrt(A ** 2 + B ** 2)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist

    def left(self, step):
        action = [[0], [0]]
        if step < 40:
            action = [[0], [30]]
        elif step < 47:
            action = [[200], [0]]
        elif step < 59:
            action = [[-96], [0]]
        elif step < 62:
            action = [[0], [-30]]
        return action


    def right(self, step):
        action = [[0], [0]]
        if step < 40:
            action = [[0], [-30]]
        elif step < 47:
            action = [[200], [0]]
        elif step < 59:
            action = [[-96], [0]]
        elif step < 62:
            action = [[0], [30]]
        return action


    def back(self, step):
        action = [[0], [0]]
        if step < 49:
            action = [[-99], [0]]
        elif step < 54:
            action = [[200], [0]]
        return action


    def get_action(self, step):
        action = [[0], [0]]
        if step < 7:
            action = [np.array([200]), np.array([0])]
        elif 7 < step < 32:
            action = [np.array([-100]), np.array([0])]
        elif step < 37:
            action = [[190], [0]]
        elif step < 79:
            if self.pos == 0:
                action = [[36], [0]]
            elif self.pos == 2:
                action = self.back(step)
            elif self.pos == 1:
                if step < 54:
                    action = self.back(step)
                else:
                    action = self.left(step - 17)
            elif self.pos == 3:
                if step < 54:
                    action = self.back(step)
                else:
                    action = self.right(step - 17)
        else:
            if self.angle > 30:
                action = [[0], [30]]
                self.angle = self.angle -30
                return action
            elif self.angle < -30 :
                action = [[0], [-30]]
                self.angle = self.angle + 30
                return action
            else:
                action = [[200], [self.angle]]
                self.angle = 0
                return action

        return action


def fixed(step):
    if step < 7:
        action = [np.array([200]), np.array([0])]
    elif 7 < step < 32:
        action = [np.array([-100]), np.array([0])]
    elif step < 37:
        action = [[190], [0]]
    return action

def run_one_actor(index, args, unknown_args, actor_status):
    import tensorflow.compat.v1 as tf
    from tensorflow.keras.backend import set_session

    # Set 'allow_growth'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Connect to learner
    context = zmq.Context()
    context.linger = 0  # For removing linger behavior
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.data_port}')

    # Initialize environment and model instance
    env_type  = 'olympics-curling'
    env = make(env_type, seed=None)
    model = get_model(args)

    # Configure logging in all process
    log_path = args.exp_path / ('log%d'%index)
    log_path.mkdir()
    logger.configure(str(log_path))


    # Initialize values
    model_id = -1
    episode_infos = deque(maxlen=100)
    num_episode = 0
    
    model_init_flag = 0
    while model_init_flag == 0:
        new_weights, model_id = find_new_weights(model_id, args.ckpt_path)
        if new_weights is not None:
            model.set_weights(new_weights)
            model_init_flag = 1

    my_agent = fix_law()
    for update in range(1, args.max_episodes):
        # Update weights
        new_weights, model_id = find_new_weights(model_id, args.ckpt_path)
        if new_weights is not None:
            model.set_weights(new_weights)
        
        obs = env.reset()
        # Collect data
        mb_states, mb_actions, mb_rewards,  mb_extras = [], [], [], []

        start_time = time.time()
        steps = 0
        lef_throws = [4, 4]
        is_trainable = 0
        last_round = 0
        
        while True:
            if is_trainable < 0:
                action0 = [np.array([0]), np.array([0])]
                action1 = [np.array([0]), np.array([0])]
                obs, reward, done, _, _, is_trainable,min_dist = env.step([action0, action1])
                if reward[0] != 0:
                    mb_rewards[-1][0] = reward[0]/10
            elif is_trainable == 0:
                if lef_throws[is_trainable] != obs[0]['throws left'][is_trainable]:
                    steps = 0
                    lef_throws[is_trainable] = obs[0]['throws left'][is_trainable]
                
                if steps < 37:
                    if steps == 22:
                        pro = Obs_Process(obs[is_trainable])
                        team, enemy, center = pro.obs_pre_process()
                        team = np.concatenate(team)
                        team = np.expand_dims(team, axis=0)
                        enemy = np.concatenate(enemy)
                        enemy = np.expand_dims(enemy, axis=0)
                        center = np.expand_dims(center, axis=0)
                    
                    real_action = fixed(steps)
                    obs, reward, done, _, _, is_trainable, min_dist = env.step([real_action, real_action])
                    steps += 1
                elif steps == 37:
                    me = np.array([[0, -0.66]])
                    if obs[is_trainable]['game round'] == obs[is_trainable]['controlled_player_index']:
                        first_to_move = [1]
                    else:
                        first_to_move = [0]
                    first_to_move = np.expand_dims(first_to_move, axis=0)
                    tl = np.expand_dims([lef_throws[is_trainable]], axis=0)

                    all_state = np.concatenate([me, team, enemy, center, first_to_move, tl], axis=1)
                    assert all_state.shape == (1, 20)
                    action, value, neglogp = model.forward(all_state)
                    real_action = [np.array([action[0][0] * 4 + 10]),np.array([action[0][1] * 2 + -28])]
                    extra_data = {'value': value, 'neglogp': neglogp}
                    mb_states.append(all_state)
                    mb_actions.append(action)
                    mb_rewards.append([0])
                    mb_extras.append(extra_data)

                    obs, reward, done, _, _, is_trainable, min_dist = env.step([real_action, real_action])
                    steps += 1
                else:
                    real_action = [[mb_actions[-1][0][0] * 4 + 10], [0]]
                          
                    obs, reward, done, _, _, is_trainable, min_dist = env.step([real_action, real_action])
                    steps += 1
            
            elif is_trainable == 1:
                if lef_throws[is_trainable] != obs[0]['throws left'][is_trainable]:
                    steps = 0
                    lef_throws[is_trainable] = obs[0]['throws left'][is_trainable]

                if steps == 22:
                    pro = Obs_Process(obs[is_trainable])
                    team, enemy, center = pro.obs_pre_process()
                    team = np.concatenate(team)
                    team = np.expand_dims(team, axis=0)
                    enemy = np.concatenate(enemy)
                    enemy = np.expand_dims(enemy, axis=0)
                    center = np.expand_dims(center, axis=0)
                    my_agent.set(team=team, enemy=enemy, center=center)

                real_action = my_agent.get_action(steps)
                obs, reward, done, _, _, is_trainable, min_dist = env.step([real_action, real_action])
                steps += 1

            if done:
                num_episode += 1

                mb_states  =  np.asarray(mb_states, dtype=np.float32)
                mb_rewards =  np.asarray(mb_rewards, dtype=np.float32)
                mb_actions =  np.asarray(mb_actions)

                data = prepare_training_data([mb_states, mb_actions, mb_rewards, mb_extras])
                socket.send(serialize(data).to_buffer())
                socket.recv()

                send_data_interval = time.time() - start_time
                # Log information
                reward_sum = 0
                for reward in mb_rewards:
                    reward_sum += reward[0]
                logger.record_tabular("episodes", num_episode)
                logger.record_tabular('first_to_move', (num_episode+1)%2)
                logger.record_tabular("min_dist", min_dist)
                logger.record_tabular("model_reward_sum", reward_sum)

                logger.dump_tabular()
                break

            cur_round = obs[0]['game round']
            if last_round != cur_round:
                last_round = cur_round
                num_episode += 1

                mb_states  =  np.asarray(mb_states, dtype=np.float32)
                mb_rewards =  np.asarray(mb_rewards, dtype=np.float32)
                mb_actions =  np.asarray(mb_actions)
           
                data = prepare_training_data([mb_states, mb_actions, mb_rewards, mb_extras])
                socket.send(serialize(data).to_buffer())
                socket.recv()

                send_data_interval = time.time() - start_time
                # Log information
                reward_sum = 0
                for reward in mb_rewards:
                    reward_sum += reward[0]
                logger.record_tabular("episodes", num_episode)
                logger.record_tabular('first_to_move', (num_episode+1)%2)
                logger.record_tabular("min_dist", min_dist)
                logger.record_tabular("model_reward_sum", reward_sum)
                logger.dump_tabular()
                mb_states, mb_actions, mb_rewards,  mb_extras = [], [], [], []


    actor_status[index] = 1


def run_weights_subscriber(args, actor_status):
    """Subscribe weights from Learner and save them locally"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything

    for model_id in count(1):  # Starts from 1
        while True:
            try:
                weights = socket.recv(flags=zmq.NOBLOCK)

                # Weights received
                with open(args.ckpt_path / f'{model_id}.{args.alg}.{args.env}.ckpt', 'wb') as f:
                    f.write(weights)

                if model_id > args.num_saved_ckpt:
                    os.remove(args.ckpt_path / f'{model_id - args.num_saved_ckpt}.{args.alg}.{args.env}.ckpt')
                break
            except zmq.Again:
                pass

            if all(actor_status):
                # All actors finished works
                return

            # For not cpu-intensive
            time.sleep(1)


def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Load config file
    load_yaml_config(args, 'actor')

    # Create experiment directory
    create_experiment_dir(args, 'ACTOR-')

    args.ckpt_path = args.exp_path / 'ckpt'
    # args.log_path = args.exp_path / 'log'
    args.ckpt_path.mkdir()
    # args.log_path.mkdir()

    # Record commit hash
    # with open(args.exp_path / 'hash', 'w') as f:
    #     f.write(str(subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')))

    # Disable GPU
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Running status of actors
    actor_status = Array('i', [0] * args.num_replicas)

    # Run weights subscriber
    subscriber = Process(target=run_weights_subscriber, args=(args, actor_status))
    subscriber.start()

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_actor(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(actors):
                    if _i != index:
                        _p.terminate()
                    actor_status[_i] = 1

    actors = []
    for i in range(args.num_replicas):
        p = Process(target=exit_wrapper, args=(i, args, unknown_args, actor_status))
        p.start()
        os.system(f'taskset -p -c {(i+0) % os.cpu_count()} {p.pid}')  # For CPU affinity

        actors.append(p)

    for actor in actors:
        actor.join()

    subscriber.join()


if __name__ == '__main__':
    main()
