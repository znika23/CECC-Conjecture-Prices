import Memory
import copy
import time
import random
import numpy as np
from collections import deque
from collections import defaultdict


# noinspection NonAsciiCharacters
# d => 需求 r => 存储
# T => 时间 M => 成本 ρ => 单价
# a => 资源购买量 g => 资源获取量 t => 任务分配量
class 边缘节点类:
    def __init__(self, config):
        # 静态数据
        self.节点: dict = {
            'A1': config['cheap_action'],
            'A2': config['costly_action'],
            'd': config['demand'],
            'r': config['storage'],
            'r_max': np.max(config['storage']),  # 计算型资源最大存储量
            'f_local': config['local_amount'],
            'ρ_local': config['local_price'],
            't_avg': config['task_size'],  # 平均任务所需计算资源量
            'l': config['time_delay'],  # 平均任务通讯时延
        }
        # 动态数据
        self.状态: dict = {
            'ρ': int,
            'd': int,
            'r': int
        }
        # 超参数
        self.γ = config['gamma']
        self.σ = config['learning_rate']
        self.batch = config['batch']
        self.memory = Memory.经验回放池类(config['memory_size'], config['batch'])
        # 策略与Q表
        self.T = config['t']  # 环境分布异常检测区间
        self.Q_Table = defaultdict(lambda: defaultdict(lambda: 0))
        self.Q_Table_Alt = defaultdict(lambda: defaultdict(lambda: 0))
        self.Policy_Table = defaultdict(int)
        self.Last_Policy_Table = defaultdict(int)
        self.Goal_Table = defaultdict(lambda: deque(maxlen=self.T))

    def 状态初始化(self, ρ):
        for table in [self.Q_Table, self.Q_Table_Alt, self.Policy_Table, self.Last_Policy_Table, self.Goal_Table]:
            table.clear()
        self.状态['ρ'] = ρ
        self.状态['d'] = self.节点['d'][random.randint(0, len(self.节点['d']) - 1)]
        self.状态['r'] = 0

    def 状态更新(self, ρ, d, r):
        self.状态['ρ'] = ρ
        self.状态['d'] = d
        self.状态['r'] = r

    def 生成动作(self):
        a1 = self.节点['A1'][random.randint(0, len(self.节点['A1']) - 1)]
        a2 = self.节点['A2'][random.randint(0, len(self.节点['A2']) - 1)]
        return a1, a2

    def 存储样本(self, 样本数据):
        ρ, d, r, a1, a2, R = 样本数据
        self.memory.add((self.状态['ρ'], self.状态['d'], self.状态['r']),
                        (ρ, d, r), (a1, a2), R)
        # print('s:', self.状态['ρ'], self.状态['d'], self.状态['r'], ' a:', a1, a2, 'r:', R, 's‘:', ρ, d, r)

    def 生成任务和样本(self, g1, g2, rng=0):
        if rng != 0:
            index = rng.integers(0, len(self.节点['d']))
            d = self.节点['d'][index]
        else:
            d = self.节点['d'][random.randint(0, len(self.节点['d']) - 1)]
        if self.状态['d'] <= g1:
            r = min(g2 + self.状态['r'], self.节点['r_max'])
            return self.状态['d'], 0, d, r
        else:
            r = self.状态['r'] + g1 + g2 - self.状态['d']
            r = min(r, self.节点['r_max'])
            r = max(r, 0)
            return g1, min(self.状态['d'] - g1, g2 + self.状态['r']), d, r

    # t_cloud : 云端任务量
    def 计算本地时间和成本(self, t_cloud):
        T_local, M_local = 0, 0
        t3 = self.状态['d'] - t_cloud
        if t3 > 0:
            T_local = t3 / self.节点['f_local']
            M_local = T_local * self.节点['ρ_local']
        # print(self.节点['ρ_local'])
        return T_local, M_local

    # T_com : 总计算时间
    # t_cloud : 云端任务量
    def 计算单位任务时间(self, T_com, t_cloud):
        # T_tran 总传输时间
        T_tran = t_cloud / self.节点['t_avg'] * self.节点['l']
        return np.round((T_com * 3600 + T_tran) / self.状态['d'] * self.节点['t_avg'], 2)

    def 计算价格差值(self, 推测价格, g_cloud, t_cloud, 总成本):
        if 总成本 == 0:
            return -推测价格
        t3 = self.状态['d'] - t_cloud
        实际价格 = np.round(总成本 / (g_cloud / 10 + t3 / self.节点['f_local']),2)
        # print('g_cloud=', g_cloud, 't_cloud=', t_cloud, 't3=', t3, 'total_cost=', 总成本)
        # print('实际价格:',实际价格, 推测价格)
        return -(np.abs(实际价格 - 推测价格))

    # def 存储样本(self):
    def 更新Q表(self, sample_index):
        抽样样本 = self.memory.sample(sample_index)
        当前状态列表 = 抽样样本[0]
        下一状态列表 = 抽样样本[1]
        动作列表 = 抽样样本[2]
        奖励列表 = 抽样样本[3]
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
        # print(self.Q_Table)

    def 生成目标策略(self):
        self.Last_Policy_Table = self.Policy_Table
        for 状态, Q_line in self.Q_Table.items():
            动作 = max(Q_line, key=Q_line.get)
            self.Policy_Table[状态] = 动作
        print(dict(sorted(self.Policy_Table.items())))

    def 执行策略(self, 策略):
        动作 = [0.0, 0.0]
        match 策略:
            case 0:
                # Edge_Only
                动作 = [0.0, 0.0]
            case 1:
                # Costly_only
                动作 = [0.0, self.状态['d']]
            case 2:
                # Edge + Costly + Cheap
                动作 = self.Policy_Table[(self.状态['ρ'], self.状态['d'], self.状态['r'])]
                if type(动作) == int:
                    动作 = [0.0, 0.0]
                    动作[0], 动作[1] = self.生成动作()
            case 3:
                动作[0] = self.节点['A1'][random.randint(0, len(self.节点['A1']) - 1)]
                动作[1] = self.节点['A2'][random.randint(0, len(self.节点['A2']) - 1)]
        # print(动作[0], 动作[1])
        return 动作[0], 动作[1]

    def 统计决策目标值(self, 目标值):
        Anomaly = False
        决策目标列表 = np.array(self.Goal_Table[(self.状态['ρ'], self.状态['d'], self.状态['r'])])
        if len(决策目标列表) == self.T:
            if 目标值 > np.max(决策目标列表) or 目标值 < np.min(决策目标列表):
                Anomaly = True
        self.Goal_Table[(self.状态['ρ'], self.状态['d'], self.状态['r'])].append(目标值)
        return Anomaly
