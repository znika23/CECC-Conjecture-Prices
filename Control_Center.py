import Cloud_Server
import Edge_Server
import Simulate_Node
import ast
import copy
import time
import pickle
import keyboard
import configparser
import numpy as np
from collections import deque
from collections import defaultdict


def convert(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            try:
                return ast.literal_eval(value)
            except ValueError:
                return value


# d => 需求 r => 存储
# T => 时间 M => 成本 ρ => 单价
# a => 资源购买量 g => 资源获取量 t => 任务分配量

# noinspection NonAsciiCharacters
class 控制中心类:
    def __init__(self):
        self.参数: dict = {}
        self.云节点 = None
        self.边缘节点列表 = None
        self.模拟节点列表 = None
        self.rng = np.random.default_rng(seed=93)
        # print(type(self.rng))
        # 检测环境分布改变参数
        self.α = None  # P(发出警告|环境分布改变)
        self.β = None  # P(发出警告|环境分布未改变)
        self.T = None
        self.分布异常检测矩阵 = None
        self.配置()
        self.初始化()

    def 配置(self):
        config = configparser.ConfigParser()
        config.read('config/config_1.ini')
        for section in config.sections():
            self.参数[section] = {key: convert(value) for key, value in config.items(section)}

    def 初始化(self):
        边缘节点数量 = len(self.参数['cloud']['edge_server'])
        self.云节点 = Cloud_Server.云节点类(self.参数['cloud'])
        self.模拟节点列表 = np.empty(边缘节点数量, dtype=object)
        self.边缘节点列表 = np.empty(边缘节点数量, dtype=object)
        for i in range(边缘节点数量):
            self.模拟节点列表[i] = Simulate_Node.模拟节点类(self.参数['simulator'])
            self.边缘节点列表[i] = Edge_Server.边缘节点类({**self.参数['edge'], **self.参数['edge' + str(i + 1)]})

    def 模拟云节点状态分布改变(self):
        self.云节点.状态分布改变()

    def 生成样本(self):
        print('-------------------构造样本-------------------')
        # 参数准备
        边缘节点数量 = len(self.参数['cloud']['edge_server'])
        # 生成样本---(s',a,R)
        # q, r_left, ρ, R_si
        模拟节点样本 = np.zeros((边缘节点数量, 4))
        # ρ, d, r, a1, a2, R_edge
        边缘节点样本 = np.zeros((边缘节点数量, 6))

        # 状态初始化
        self.云节点.状态初始化(self.rng)
        for i in range(边缘节点数量):
            self.模拟节点列表[i].状态初始化(self.云节点.状态['q'])
            ρ = self.模拟节点列表[i].生成动作()
            self.边缘节点列表[i].状态初始化(ρ)
            模拟节点样本[i, 2] = ρ

        # 构造样本
        for 样本个数 in range(self.参数['center']['max_sample_size'] - 1):
            if 样本个数 % 5000 == 0:
                print('样本构造中:', np.round(样本个数 / self.参数['center']['max_sample_size'] * 100, 2), '%')
            #
            for i in range(边缘节点数量):
                a1, a2 = self.边缘节点列表[i].生成动作()
                边缘节点样本[i, 3] = a1
                边缘节点样本[i, 4] = a2
            # 资源分配
            # g1 经济型资源实际获取量
            # g2 计算型资源实际获取量
            g1 = self.云节点.经济型资源分配(边缘节点样本[:, 3])
            g2 = 边缘节点样本[:, 4]
            # 任务列表
            t1 = np.zeros(边缘节点数量)
            t2 = np.zeros(边缘节点数量)
            # 任务分配
            for i in range(边缘节点数量):
                t1[i], t2[i], d, r = self.边缘节点列表[i].生成任务和样本(g1[i], g2[i])
                边缘节点样本[i, 1] = d
                边缘节点样本[i, 2] = r
                模拟节点样本[i, 1] = g1[i] + g2[i] - t1[i] - t2[i]
            # reward计算
            成本列表 = self.云节点.云端成本计算(g1, g2)
            时间列表 = self.云节点.云端时间计算(g1, t1, t2)

            for i in range(边缘节点数量):
                本地时间, 本地成本 = self.边缘节点列表[i].计算本地时间和成本(t1[i] + t2[i])
                时间列表[i] += 本地时间
                成本列表[i] += 本地成本
                时间列表[i] = self.边缘节点列表[i].计算单位任务时间(np.sum(时间列表[i]), t1[i] + t2[i])
                R_si = self.边缘节点列表[i].计算价格差值(模拟节点样本[i, 2], g1[i] + g2[i], t1[i] + t2[i], 成本列表[i])
                R_edge = np.round(
                    - self.参数['center']['time_ratio'] * 时间列表[i]
                    - self.参数['center']['cost_ratio'] * 成本列表[i], 2
                )
                模拟节点样本[i, 3] = R_si
                边缘节点样本[i, 5] = R_edge
            # print('-------样本---------------')
            # print(边缘节点样本[3])

            # 存储样本 + 状态更新
            self.云节点.状态更新(self.rng)
            模拟节点样本[:, 0] = self.云节点.状态['q']
            for i in range(边缘节点数量):
                self.模拟节点列表[i].存储样本(模拟节点样本[i])
                self.模拟节点列表[i].状态更新(模拟节点样本[i, 0], 模拟节点样本[i, 1])
                ρ = self.模拟节点列表[i].生成动作()
                边缘节点样本[i, 0] = ρ
                self.边缘节点列表[i].存储样本(边缘节点样本[i])
                self.边缘节点列表[i].状态更新(
                    边缘节点样本[i, 0], 边缘节点样本[i, 1], 边缘节点样本[i, 2]
                )
                模拟节点样本[i, 2] = ρ

    def 学习(self):
        # 学习
        print('-------------------开始学习-------------------')
        episode = 0
        边缘节点数量 = len(self.参数['cloud']['edge_server'])
        收敛曲线 = np.zeros((3, self.参数['center']['max_train_round']))
        while episode < self.参数['center']['max_train_round']:
            print('当前轮次:', episode)
            sample_index = np.random.choice(self.参数['center']['max_sample_size'] - 1, self.参数['edge']['batch'],
                                            replace=False)
            for i in range(边缘节点数量):
                self.模拟节点列表[i].更新Q表(sample_index)
                self.模拟节点列表[i].生成目标策略()
                self.边缘节点列表[i].更新Q表(sample_index)
                self.边缘节点列表[i].生成目标策略()
            r_si, total_cost, total_time = self.模拟上线()
            episode += 1
        for i in range(边缘节点数量):
            print('Policy_Manager_{}'.format(i + 1), ' = ', dict(sorted(self.模拟节点列表[i].Policy_Table.items())))
            print('Policy_Edge_{}'.format(i + 1), ' = ', dict(sorted(self.边缘节点列表[i].Policy_Table.items())))

    def 模拟上线(self):
        # 参数准备
        边缘节点数量 = len(self.参数['cloud']['edge_server'])
        # q, r_left, ρ
        模拟节点数据 = np.zeros((边缘节点数量, 3))
        # ρ, d, r, a1, a2
        边缘节点数据 = np.zeros((边缘节点数量, 5))
        #
        统计列表 = np.zeros(3)

        # 初始状态更新
        for i in range(边缘节点数量):
            self.模拟节点列表[i].状态更新(self.云节点.状态['q'], 0)
            ρ = self.模拟节点列表[i].执行策略()
            self.边缘节点列表[i].状态更新(ρ, 40, 0)
            模拟节点数据[i, 2] = ρ

        print('-------------------模拟上线-------------------')
        episode = 0
        while episode < self.参数['center']['max_online_round']:
            for i in range(边缘节点数量):
                a1, a2 = self.边缘节点列表[i].执行策略(2)
                边缘节点数据[i, 3] = a1
                边缘节点数据[i, 4] = a2
            # 资源分配
            g1 = self.云节点.经济型资源分配(边缘节点数据[:, 3])
            g2 = 边缘节点数据[:, 4]
            # 任务分配
            t1 = np.zeros(边缘节点数量)
            t2 = np.zeros(边缘节点数量)
            for i in range(边缘节点数量):
                t1[i], t2[i], d, r = self.边缘节点列表[i].生成任务和样本(g1[i], g2[i])
                边缘节点数据[i, 1] = d
                边缘节点数据[i, 2] = r
                模拟节点数据[i, 1] = g1[i] + g2[i] - t1[i] - t2[i]
            # reward计算
            成本列表 = self.云节点.云端成本计算(g1, g2)
            时间列表 = self.云节点.云端时间计算(g1, t1, t2)
            for i in range(边缘节点数量):
                本地时间, 本地成本 = self.边缘节点列表[i].计算本地时间和成本(t1[i] + t2[i])
                时间列表[i] += 本地时间
                成本列表[i] += 本地成本
                时间列表[i] = self.边缘节点列表[i].计算单位任务时间(np.sum(时间列表[i]), t1[i] + t2[i])
                R_si = self.边缘节点列表[i].计算价格差值(模拟节点数据[i, 2], g1[i] + g2[i], t1[i] + t2[i], 成本列表[i])
                统计列表[0] += R_si
                统计列表[1] += 成本列表[i]
                统计列表[2] += 时间列表[i]

            # 状态更新
            self.云节点.状态更新(self.rng)
            模拟节点数据[:, 0] = self.云节点.状态['q']
            for i in range(边缘节点数量):
                self.模拟节点列表[i].状态更新(模拟节点数据[i, 0], 模拟节点数据[i, 1])
                ρ = self.模拟节点列表[i].执行策略()
                边缘节点数据[i, 0] = ρ
                self.边缘节点列表[i].状态更新(
                    边缘节点数据[i, 0], 边缘节点数据[i, 1], 边缘节点数据[i, 2]
                )
                模拟节点数据[i, 2] = ρ
            episode += 1
        print('累计R_si为：', np.round(统计列表[0], 2),
              '累计成本为:', np.round(统计列表[1], 2),
              '累计时间为:', np.round(统计列表[2], 2))
        return 统计列表[0], 统计列表[1], 统计列表[2]

    def 实际上线(self):
        # 参数准备
        边缘节点数量 = len(self.参数['cloud']['edge_server'])
        # 节点数据
        模拟节点数据 = np.zeros((边缘节点数量, 3))  # q, r_left, ρ
        边缘节点数据 = np.zeros((边缘节点数量, 5))  # ρ, d, r, a1, a2
        统计列表 = np.zeros(3)
        #
        g1_acc = np.zeros(self.参数['center']['max_online_round'])
        g2_acc = np.zeros(self.参数['center']['max_online_round'])
        g3_acc = np.zeros(self.参数['center']['max_online_round'])
        time_acc = np.zeros(self.参数['center']['max_online_round'])
        cost_acc = np.zeros(self.参数['center']['max_online_round'])

        # 初始状态更新
        for i in range(边缘节点数量):
            self.模拟节点列表[i].状态更新(self.云节点.状态['q'], 0)
            ρ = self.模拟节点列表[i].执行策略()
            self.边缘节点列表[i].状态更新(ρ, 40, 0)
            模拟节点数据[i, 2] = ρ

        g = np.zeros((4, 3))

        print('-------------------实际上线-------------------')
        episode = 0
        while episode < self.参数['center']['max_online_round']:
            for i in range(边缘节点数量):
                a1, a2 = self.边缘节点列表[i].执行策略(3)
                边缘节点数据[i, 3] = a1
                边缘节点数据[i, 4] = a2
            # 资源分配
            g1 = self.云节点.经济型资源分配(边缘节点数据[:, 3])
            g2 = 边缘节点数据[:, 4]
            # 任务分配
            t1 = np.zeros(边缘节点数量)
            t2 = np.zeros(边缘节点数量)
            for i in range(边缘节点数量):
                t1[i], t2[i], d, r = self.边缘节点列表[i].生成任务和样本(g1[i], g2[i], self.rng)
                边缘节点数据[i, 1] = d
                边缘节点数据[i, 2] = r
                模拟节点数据[i, 1] = g1[i] + g2[i] - t1[i] - t2[i]
                g[i, 0] += g1[i]
                g[i, 1] += g2[i]
            g1_acc[episode] = g1[3]
            g2_acc[episode] = g2[3]
            成本列表 = self.云节点.云端成本计算(g1, g2)
            时间列表 = self.云节点.云端时间计算(g1, t1, t2)
            for i in range(边缘节点数量):
                # reward计算
                本地时间, 本地成本 = self.边缘节点列表[i].计算本地时间和成本(t1[i] + t2[i])
                时间列表[i] += 本地时间
                成本列表[i] += 本地成本
                时间列表[i] = self.边缘节点列表[i].计算单位任务时间(np.sum(时间列表[i]), t1[i] + t2[i])
                R_si = self.边缘节点列表[i].计算价格差值(模拟节点数据[i, 2], g1[i] + g2[i], t1[i] + t2[i], 成本列表[i])
                R_edge = np.round(
                    - self.参数['center']['time_ratio'] * 时间列表[i]
                    - self.参数['center']['cost_ratio'] * 成本列表[i], 2
                )
                统计列表[0] += R_si
                统计列表[1] += 成本列表[i]
                统计列表[2] += 时间列表[i]
            g3 = 0
            d = 0
            g3_acc[episode] = self.边缘节点列表[3].状态['d'] - t1[3] - t2[3]
            for i in range(边缘节点数量):
                g3 += self.边缘节点列表[i].状态['d'] - t1[i] - t2[i]
                g[i, 2] += self.边缘节点列表[i].状态['d'] - t1[i] - t2[i]
                d += self.边缘节点列表[i].状态['d']
            print('-----------------', episode + 1, '----------------')
            print('经济型资源量:', self.云节点.状态['q'])
            print('总需求：', d)
            print('获得资源量:', np.sum(g1), np.sum(g2), g3)
            # 状态更新
            self.云节点.状态更新(self.rng)
            模拟节点数据[:, 0] = self.云节点.状态['q']
            for i in range(边缘节点数量):
                self.模拟节点列表[i].状态更新(模拟节点数据[i, 0], 模拟节点数据[i, 1])
                ρ = self.模拟节点列表[i].执行策略()
                边缘节点数据[i, 0] = ρ
                self.边缘节点列表[i].状态更新(
                    边缘节点数据[i, 0], 边缘节点数据[i, 1], 边缘节点数据[i, 2]
                )
                模拟节点数据[i, 2] = ρ
            cost_acc[episode] = np.round(统计列表[1], 2)
            time_acc[episode] = np.round(统计列表[2], 2)
            print('累计成本为:', np.round(统计列表[1], 2),
                  '累计时间为:', np.round(统计列表[2], 2))
            episode += 1
        local = 0
        total = 0
        for i in range(len(g)):
            local += g[i, 2]
            total += np.sum(g[i])
        print(local / total)
        print('累计资源购买量：', g)
        # np.save('data/strategy_acc/g1_random.npy', g1_acc)
        # np.save('data/strategy_acc/g2_random.npy', g2_acc)
        # np.save('data/strategy_acc/g3_random.npy', g3_acc)
        # np.save('data/strategy_acc/cost_random.npy', cost_acc)
        # np.save('data/strategy_acc/time_random.npy', time_acc)

        # np.save('data/strategy_acc/g1_optimal.npy', g1_acc)
        # np.save('data/strategy_acc/g2_optimal.npy', g2_acc)
        # np.save('data/strategy_acc/g3_optimal.npy', g3_acc)
        # np.save('data/strategy_acc/cost_optimal.npy', cost_acc)
        # np.save('data/strategy_acc/time_optimal.npy', time_acc)

    def 读取策略(self):
        边缘节点数量 = len(self.参数['cloud']['edge_server'])
        self.云节点.状态初始化(self.rng)
        for i in range(边缘节点数量):
            self.模拟节点列表[i].状态初始化(self.云节点.状态['q'])
            with open("data/manager_policy/edge_{}_{}".format(i + 1, 'config_1_test') + '.pkl', 'rb') as f:
                self.模拟节点列表[i].Policy_Table = pickle.load(f)
            ρ = self.模拟节点列表[i].执行策略()
            self.边缘节点列表[i].状态初始化(ρ)
            with open("data/edge_policy/edge_{}_{}".format(i + 1, 'config_1_test') + '.pkl', 'rb') as f:
                self.边缘节点列表[i].Policy_Table = pickle.load(f)
