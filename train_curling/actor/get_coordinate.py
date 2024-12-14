import math
import numpy as np
import copy


Bound = 60

class Obs_Process():
    def __init__(self, obs):
        # 冰壶在矩阵中坐标
        self.fix_x = 30.5
        self.fix_y = 14.5
        self.red = 7
        self.grey = 4
        self.obs = obs
        # 看不见基准线时返回的坐标
        self.min_cor = -60
        self.image = self.obs['obs'][0]
        self.rho = 1
        self.points = 15

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
                        # print(pic[i-1:i+2, j-1:j+2])
                        pic[i - 1:i + 2, j - 1:j + 2] = 0
                        # print([j, i])
                        result.append([j, i])

        return result

    def obs_pre_process(self):
        '''
        返回冰壶坐标
        '''
        pic = copy.deepcopy(self.obs['obs'][0])
        origin = np.array([14.45, 25])
        center = np.array([14.45, 10])

        # 找到�?
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
        center = (origin - center)/20
        enemy = [np.array([(item[0] - origin[0])/20, (origin[1] - item[1])/20]) for item in enemy]
        team = [np.array([(item[0] - origin[0])/20, (origin[1] - item[1])/20]) for item in team]
        while len(team) < 3:
            team.append(np.array([-3, -3]))
        while len(enemy) < 4:
            enemy.append(np.array([-3, -3]))

        return team, enemy, center
