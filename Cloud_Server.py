import copy
import random
import numpy as np


# noinspection NonAsciiCharacters
class 云节点类:
    def __init__(self, config):
        # 静态数据
        self.节点: dict = {
            '边缘节点': config['edge_server'],
            'U': config['cheap_distribution'],  # 经济型计算资源总量取值分布([CPU GCycles,...,CPU GCycles])
            'f_e': config['cheap_amount'],  # 经济型云服务器频率(CPU GCycles/Hour)
            'ρ_e': config['cheap_price'],  # 经济型每小时单价(RMB/Hour)
            'f_c': config['costly_amount'],  # 计算型云服务器频率(CPU GCycles/Hour)
            'ρ_c': config['costly_price'],  # 计算型每小时单价(RMB/Hour)
        }
        # 动态数据
        self.状态: dict = {
            'q': int,
            'U': None
        }
        self.count = 0
        self.分布 = [[40, 100, 160], [0, 80, 160]]

    # def 状态初始化(self):
    #     self.状态['U'] = sorted(random.sample(self.节点['U'], k=3))
    #     # self.状态['U'] = self.分布[self.count]
    #     # self.count += 1
    #     # self.count %= 2
    #     # todo
    #     self.状态['U'] = [20, 80, 140]
    #     print('实时经济型分布', self.状态['U'])
    #     self.状态更新()

    # def 状态更新(self):
    #     index = random.randint(0, len(self.状态['U']) - 1)
    #     self.状态['q'] = self.状态['U'][index]
    #     print(self.状态['q'])

    def 状态初始化(self, rng):
        self.状态['U'] = [20, 80, 140]
        print('实时经济型分布', self.状态['U'])
        self.状态更新(rng)

    def 状态更新(self, rng):
        index = rng.integers(0, len(self.状态['U']))
        self.状态['q'] = self.状态['U'][index]

    def 经济型资源分配(self, 购买列表):
        if np.sum(购买列表) <= self.状态['q']:
            return 购买列表
        elif self.状态['q'] == 0:
            return np.zeros(len(购买列表))
        经济型购买列表 = copy.copy(购买列表)
        平均分配单位 = self.状态['q'] / np.sum(经济型购买列表)
        经济型购买列表 *= 平均分配单位
        经济型分配列表 = np.array(
            [int(购买量 / self.节点['f_e']) * self.节点['f_e'] for 购买量 in 经济型购买列表])
        还需分配经济型单位 = int((self.状态['q'] - np.sum(经济型分配列表)) / self.节点['f_e'])
        index = np.argsort(经济型购买列表 - 经济型分配列表)[::-1][:还需分配经济型单位]
        经济型分配列表[index] += self.节点['f_e']
        return 经济型分配列表

    def 云端时间计算(self, g1, t1, t2):
        # 计算分配给经济型资源的任务耗时
        经济型总请求量 = np.sum(g1)
        经济型频率分配 = np.array([资源量 / 经济型总请求量 for 资源量 in g1]) * self.节点['f_e']
        时间列表 = np.divide(np.array(t1[:]), 经济型频率分配, out=np.zeros_like(经济型频率分配),
                             where=np.array(t1[:]) != 0)
        # 计算分配给计算型资源的任务耗时
        时间列表 += np.array(t2[:]) / self.节点['f_c']
        return 时间列表

    def 云端成本计算(self, 经济型资源列表, 计算型购买列表):
        # 计算经济型成本
        成本列表 = np.array([购买量 / self.节点['f_e'] * self.节点['ρ_e'] for 购买量 in 经济型资源列表])
        # 计算计算型成本
        成本列表 += [购买量 / self.节点['f_c'] * self.节点['ρ_c'] for 购买量 in 计算型购买列表]
        return 成本列表
