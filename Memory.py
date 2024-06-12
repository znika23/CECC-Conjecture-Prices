import numpy as np


# noinspection NonAsciiCharacters
class 经验回放池类:
    def __init__(self, max_len, batch_size):
        self.当前状态列表 = np.empty(max_len, dtype=object)
        self.下一状态列表 = np.empty(max_len, dtype=object)
        self.动作列表 = np.empty(max_len, dtype=object)
        self.奖励列表 = np.empty(max_len)
        self.count = 0
        self.size = max_len
        self.batch_size = batch_size

    def add(self, 当前状态, 下一状态, 动作, 奖励):
        self.当前状态列表[self.count] = 当前状态
        self.下一状态列表[self.count] = 下一状态
        self.动作列表[self.count] = 动作
        self.奖励列表[self.count] = 奖励
        self.count = (self.count + 1) % self.size

    def Sample(self):
        assert self.count >= self.batch_size
        sample_index = np.random.choice(self.count, self.batch_size, replace=False)
        batch = (
            self.当前状态列表[sample_index],
            self.下一状态列表[sample_index],
            self.动作列表[sample_index],
            self.奖励列表[sample_index]
        )
        return batch

    def sample(self, sample_index):
        assert self.count >= self.batch_size
        batch = (
            self.当前状态列表[sample_index],
            self.下一状态列表[sample_index],
            self.动作列表[sample_index],
            self.奖励列表[sample_index]
        )
        return batch
