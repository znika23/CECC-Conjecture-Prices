import copy
import json
import os.path
from tqdm import tqdm
import numpy as np
from __init__ import params_data_dir
from cloud_server import Cloud_Server
from edge_server import Edge_Server
from __init__ import samples_data_dir


# noinspection PyUnresolvedReferences, NonAsciiCharacters, PyTypeChecker
class Resource_Manager:
    DEFAULTS = {
        "N": 4,  # Nums of edge servers
    }

    def __init__(self, config):
        self.__dict__.update(Resource_Manager.DEFAULTS, **config)

        self.samples_data_dir = os.path.join(samples_data_dir, 'risk')

        self.cloud_server, self.edge_servers = self.setup(config)

    def setup(self, config):
        cloud_server = Cloud_Server({**config["general"], **config["cloud"], "is_security": False})
        edge_servers = np.empty(self.N, dtype=object)
        for n in range(self.N):
            edge_servers[n] = Edge_Server({**config["general"], **config[f"{n + 1}"], "is_security": False})
        return cloud_server, edge_servers

    def manipulate(self, policy='Rand', mode='Cache'):
        """
        Manipulate the process of CECC with data open access
        """
        # arrays for caching edge servers'
        # purchased resources a_u, a_d & obtained resources o_u, o_d & executed resources t_u, t_d
        arr_a, arr_o, arr_t = np.zeros((self.N, 2)), np.zeros((self.N, 2)), np.zeros((self.N, 2))
        # resource demand d, excess purchased universal resource p & reward r^p
        arr_d, arr_p, arr_r = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)
        #
        if mode == 'Train':
            train_r_p = np.zeros((self.general['man_ep'], self.N))
            train_M = np.zeros((self.general['man_ep'], self.N))
            train_T = np.zeros((self.general['man_ep'], self.N))

        # init
        self.cloud_server.env_change()
        for n in range(self.N):
            self.edge_servers[n].env_change()
            arr_d[n] = self.edge_servers[n].d
        arr_d_ = np.sum(arr_d) - arr_d
        for n in range(self.N):
            # q, d- is open access
            self.edge_servers[n].update_state(self.cloud_server.q, arr_p[n], arr_d_[n])

        #
        iterator = tqdm(range(self.general['man_ep'])) if mode == 'Cache' else range(self.general['man_ep'])
        for ep in iterator:
            # resource purchase
            for n in range(self.N):
                a_u, a_d = self.edge_servers[n].resource_purchase(policy)
                arr_a[n, 0], arr_a[n, 1] = a_u, a_d

            # resource allocation
            arr_o[:, 0] = arr_a[:, 0]
            arr_o[:, 1] = self.cloud_server.resource_allocation(arr_a[:, 1])

            # task schedule
            for n in range(self.N):
                arr_t[n, 0], arr_t[n, 1], arr_p[n] = self.edge_servers[n].task_schedule(arr_o[n, 0], arr_o[n, 1])

            # task execution
            # arr for caching usage cost arr_M & task time arr_T
            arr_M, arr_T = self.cloud_server.task_execution(arr_o, arr_t)
            # should be parallel in real situation
            for n in range(self.N):
                self.edge_servers[n].task_execution()

            # reward calculation
            for n in range(self.N):
                # r_p reward for resource purchase policy
                _, arr_r[n] = self.edge_servers[n].reward_calculation(arr_M[n], arr_T[n])
                if mode == 'Train':
                    train_r_p[ep, n] = arr_r[n]
                    train_M[ep, n] = self.edge_servers[n].M
                    train_T[ep, n] = self.edge_servers[n].T

            # observe next state
            self.cloud_server.env_change()
            for n in range(self.N):
                self.edge_servers[n].env_change()
                arr_d[n] = self.edge_servers[n].d
            arr_d_ = np.sum(arr_d) - arr_d
            for n in range(self.N):
                self.edge_servers[n].update_state(self.cloud_server.q, arr_p[n], arr_d_[n])

            # cache samples
            for n in range(self.N):
                self.edge_servers[n].agent.cache_sample(arr_a[n, 0], arr_a[n, 1], arr_r[n])

        # save dataset or print loss
        if mode == 'Cache':
            ξ = self.general['ξ']
            for n in range(self.N):
                self.edge_servers[n].agent.storage_sample(
                    os.path.join(self.samples_data_dir, f'{ξ}_{n + 1}_edge_risk.npy')
                )
        elif mode == 'Train':
            # print('\n', np.mean(train_r_p[:, 0]))
            # print(np.mean(train_M), np.mean(train_T))
            return np.mean(train_M), np.mean(train_T)

    def train(self):
        # load dataset
        ξ = self.general['ξ']
        for n in range(self.N):
            sample_size = self.edge_servers[n].agent.load_sample(
                os.path.join(self.samples_data_dir, f'{ξ}_{n + 1}_edge_risk.npy')
            )

        # train
        for ep in tqdm(range(self.general['train_ep'])):
            sample_index = np.random.choice(sample_size, self.general['batch'], replace=False)
            for n in range(self.N):
                self.edge_servers[n].agent.train_step(sample_index)
                self.edge_servers[n].agent.generate_policy()
            if ep % 100 == 0:
                self.manipulate('RL', 'Train')


# warnings.filterwarnings("ignore", category=RuntimeWarning)
config = os.path.join(params_data_dir, "params_train_risk.json")
with open(config, 'r', encoding='utf-8') as f:
    config = json.load(f)
resource_manager = Resource_Manager(config)

# resource_manager.manipulate()
# resource_manager.train()
