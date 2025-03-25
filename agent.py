import os
import copy
import numpy as np
from utils import ReplayBuffer
from collections import defaultdict


# noinspection PyUnresolvedReferences, NonAsciiCharacters, PyTypeChecker
class RL_Agent:
    def __init__(self, config):
        self.__dict__.update(**config)
        #
        self.Q_Table = defaultdict(lambda: defaultdict(lambda: 0.0))
        #
        self.Q_Table_Alt = defaultdict(lambda: defaultdict(lambda: 0.0))
        #
        self.memory = ReplayBuffer(self.capacity, self.batch)

        #
        self.state, self.pre_state = self.init_state(), self.init_state()
        #
        self.Policy_Table = self.init_policy_table()

    def init_state(self):
        raise NotImplementedError

    def init_policy_table(self):
        raise NotImplementedError

    def update_state(self, updates: dict):
        self.pre_state = self.state.copy()
        for key, value in updates.items():
            if key in self.state:
                self.state[key] = value
            else:
                raise KeyError(f"state {key} not exists.")

    def get_state(self, key=None):
        if key is None:
            return list(self.pre_state.values()), list(self.state.values())
        else:
            if key in self.state:
                return self.state[key]
            else:
                return 0

    def clear(self):
        for table in [self.Q_Table, self.Q_Table_Alt, self.Policy_Table]:
            table.clear()

    def train_step(self, sample_index):
        # load batch samples
        samples = self.memory.sample(sample_index)
        dims, dima = self.dims, self.dima
        γ, σ = self.γ, self.σ

        # split to trajectory
        S = list(map(tuple, samples[:, :dims]))
        if dima == 1:
            A = samples[:, dims]
        else:
            A = list(map(tuple, samples[:, dims:dims + dima]))
        R = samples[:, dims + dima]
        S_ = list(map(tuple, samples[:, dims + dima + 1:]))

        # Q_Learning: TD-error = (r + γ * max Q(s',a') - Q(s,a))
        for i in range(len(samples)):
            try:
                max_key = max(self.Q_Table[S_[i]], key=self.Q_Table[S_[i]].get)
                self.Q_Table_Alt[S[i]][A[i]] = np.round(
                    (1 - σ) * self.Q_Table[S[i]][A[i]] + σ * (R[i] + γ * self.Q_Table[S_[i]][max_key]), 2
                )
            except ValueError:
                self.Q_Table_Alt[S[i]][A[i]] = np.round(
                    (1 - σ) * self.Q_Table[S[i]][A[i]] + σ * R[i], 2
                )
        self.Q_Table = copy.deepcopy(self.Q_Table_Alt)

    def generate_policy(self):
        for s, Q_line in self.Q_Table.items():
            a = max(Q_line, key=Q_line.get)
            self.Policy_Table[s] = a

    def storage_sample(self, file_path):
        if os.path.exists(file_path):
            updated_data = np.vstack((np.load(file_path), self.memory.data))
            np.save(file_path, updated_data)
        else:
            np.save(file_path, self.memory.data)

    def load_sample(self, file_path):
        self.memory.load_from_npy(file_path)
        return len(self.memory)


class Edge_Agent(RL_Agent):
    """
    Edge server agents to get optimum resource purchase strategy
    """

    def init_state(self):
        return {
            "ρ_c": 0,  # conjecture price (ρ_c)
            "d": 0,  # resource demand  (d)
            "p": 0,  # amount of excess purchased universal resource (p)
        }

    def init_policy_table(self):
        return defaultdict(lambda: (-1, -1))

    def generate_action(self):
        ρ_c, d, p = self.state['ρ_c'], self.state['d'], self.state['p']
        return self.Policy_Table[(ρ_c, d, p)]

    def cache_sample(self, a_u, a_o, r_p):
        pre_state, state = self.get_state()
        _sample = (pre_state, a_u, a_o, r_p, state)
        self.memory.add(_sample)


class Edge_Agent_Risk(RL_Agent):
    """
    Risk Edge server agents with data open access
    """

    def init_state(self):
        return {
            "q": 0,  # dynamic resource supply (q)
            "d": 0,  # personal resource demand  (d)
            "d-": 0,  # sum of other ESs' demand (d-)
            "p": 0,  # amount of excess purchased universal resource (p)
        }

    def init_policy_table(self):
        return defaultdict(lambda: (-1, -1))

    def generate_action(self):
        q, d, p, d_ = self.state['q'], self.state['d'], self.state['p'], self.state['d-']
        return self.Policy_Table[(q, d, d_, p)]

    def cache_sample(self, a_u, a_o, r_p):
        pre_state, state = self.get_state()
        _sample = (pre_state, a_u, a_o, r_p, state)
        self.memory.add(_sample)


class Cloud_Agent(RL_Agent):
    """
    Cloud server agents to get optimum conjecture prices for edge
    """

    def init_state(self):
        return {
            "q": 0,  # hourly quantity of dynamic resources
            "c": 0,  # transaction data
        }

    def init_policy_table(self):
        return defaultdict(lambda: -1)

    def generate_action(self):
        q, c = self.state['q'], self.state['c']
        return self.Policy_Table[(q, c)]

    def cache_sample(self, ρ_c, r_c):
        pre_state, state = self.get_state()
        _sample = (pre_state, ρ_c, r_c, state)
        self.memory.add(_sample)
