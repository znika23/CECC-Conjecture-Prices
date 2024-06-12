import Memory
import random
import numpy as np
import copy
from collections import defaultdict


# noinspection NonAsciiCharacters
class 模拟节点类:
    def __init__(self, config):
        # 静态数据
        self.节点: dict = {
            'A1': config['cheap_action'],
            'A2': config['costly_action'],
            'ρ': config['conjectured_price']
        }
        # 动态数据
        self.状态: dict = {
            'q': int,
            'r_left': int
        }
        # 超参数
        self.γ = config['gamma']
        self.σ = config['learning_rate']
        self.batch = config['batch']
        self.memory = Memory.经验回放池类(config['memory_size'], config['batch'])
        # 策略与Q表
        self.Policy_Table = defaultdict(lambda: np.max(config['conjectured_price']))
        self.Q_Table = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Q_Table_Alt = defaultdict(lambda: defaultdict(lambda: 0.0))

    def 状态初始化(self, q):
        for table in [self.Q_Table, self.Q_Table_Alt, self.Policy_Table]:
            table.clear()
        self.状态['q'] = q
        self.状态['r_left'] = 0

    def 状态更新(self, q, r_left):
        self.状态['q'] = q
        self.状态['r_left'] = r_left

    def 生成动作(self):
        return self.节点['ρ'][random.randint(0, len(self.节点['ρ']) - 1)]

    def 存储样本(self, 样本数据):
        q, r_left, ρ, r = 样本数据
        self.memory.add((self.状态['q'], self.状态['r_left']),
                        (q, r_left), ρ, r)

    def 更新Q表(self, sample_index):
        抽样样本 = self.memory.sample(sample_index)
        当前状态列表 = 抽样样本[0]
        下一状态列表 = 抽样样本[1]
        动作列表 = 抽样样本[2]
        奖励列表 = 抽样样本[3]
        # print(抽样样本)
        for i in range(len(当前状态列表)):
            try:
                # Q_Learning: TD-error = (r + γ * max Q(s',a') - Q(s,a))
                max_key = max(self.Q_Table[下一状态列表[i]], key=self.Q_Table[下一状态列表[i]].get)
                self.Q_Table_Alt[当前状态列表[i]][动作列表[i]] = np.round(
                    (1 - self.σ) * self.Q_Table[当前状态列表[i]][动作列表[i]] +
                    self.σ * (奖励列表[i] + self.γ * self.Q_Table[下一状态列表[i]][max_key])
                    , 2
                )
            except ValueError:
                self.Q_Table_Alt[当前状态列表[i]][动作列表[i]] = np.round(
                    (1 - self.σ) * self.Q_Table[当前状态列表[i]][动作列表[i]] + self.σ * 奖励列表[i], 2
                )
        self.Q_Table = copy.deepcopy(self.Q_Table_Alt)

    def 生成目标策略(self):
        for 状态, Q_line in self.Q_Table.items():
            动作 = max(Q_line, key=Q_line.get)
            self.Policy_Table[状态] = 动作
        print(dict(sorted(self.Policy_Table.items())))

    def 执行策略(self):
        return self.Policy_Table[(self.状态['q'], self.状态['r_left'])]
