import copy
import numpy as np
from agent import Cloud_Agent


# noinspection PyUnresolvedReferences, NonAsciiCharacters, PyTypeChecker
class Cloud_Server:
    DEFAULTS = {
        "N": 4,  # Nums of edge servers
        "ρ_u": 3.0,  # Hourly price of universal resource (RMB)
        "f_u": 100,  # Hourly frequency of universal resource (kG CPU cycles/h)
        "ρ_d": 0.5,  # Hourly price of dynamic resource (RMB)
        "f_d": 100,  # Hourly frequency of dynamic resource (kG CPU cycles/h)
        "U": [0, 800, 1600],  # Distribution of hourly available dynamic resource (kG CPU cycles)
        "ρ_c": [1.8, 2.2, 2.6, 3.0]  # Conjecture price (RMB)
    }

    def __init__(self, config):
        self.__dict__.update(Cloud_Server.DEFAULTS, **config)

        self.rng = np.random.default_rng(self.seed)
        # available cloud dynamic resource
        self.q = 0

        # setup agents for generating conjecture prices for edge servers
        self.agents = np.empty(self.N, dtype=object)
        self.setup()

    def setup(self):
        # insecurity scenario with information keeps public -> no agent for cloud
        if not self.is_security:
            return

        # setup agents
        config = {
            "dims": 2,
            "dima": 1,
            "γ": self.γ,
            "σ": self.σ,
            "capacity": self.capacity,
            "batch": self.batch
        }
        for n in range(self.N):
            self.agents[n] = Cloud_Agent(config)

    def env_change(self):
        self.q = self.rng.choice(self.U)

    def update_state(self, arr_c):
        for n in range(self.N):
            self.agents[n].update_state({
                "q": self.q,
                "c": arr_c[n]
            })

    def conjecture_prices(self, policy='Rand'):
        """
        Generate conjecture prices for edges
        """
        arr_ρ_c = np.empty(self.N, dtype=float)
        match policy:
            case 'Rand':
                for n in range(self.N):
                    arr_ρ_c[n] = self.rng.choice(self.ρ_c)
            case 'RL':
                for n in range(self.N):
                    arr_ρ_c[n] = self.agents[n].generate_action()
                    if arr_ρ_c[n] < 0:
                        arr_ρ_c[n] = self.rng.choice(self.ρ_c)
            case _:
                for n in range(self.N):
                    arr_ρ_c[n] = self.rng.choice(self.ρ_c)
        return arr_ρ_c

    def resource_allocation(self, purchase_requests):
        """
        Allocate dynamic resources proportionally based required volume a_d
        """
        if np.sum(purchase_requests) <= self.q:
            return purchase_requests
        elif self.q == 0:
            return np.zeros(self.N)

        arr_a = copy.copy(purchase_requests)
        arr_a *= self.q / np.sum(arr_a)
        #
        arr_o = np.array([int(a / self.f_d) * self.f_d for a in arr_a])
        #
        index = np.argsort(arr_a-arr_o)[::-1][:int((self.q-np.sum(arr_o)) / self.f_d)]
        arr_o[index] += self.f_d
        return arr_o

    def task_execution(self, arr_o, arr_t):
        """
        manipulate cloud task execution based on scheduled resources
        return:
        arr_M: array for cache usage charge
        arr_T: array for cache task time
        """
        #
        arr_M = np.array([o_u / self.f_u * self.ρ_u for o_u in arr_o[:, 0]])
        arr_M += [o_d / self.f_d * self.ρ_d for o_d in arr_o[:, 1]]
        #
        arr_T = np.array(arr_t[:, 0]) / self.f_u
        sum_o_d = np.sum(arr_o[:, 1])
        arr_f_allc = np.array([o_d / sum_o_d if sum_o_d != 0 else 0.0 for o_d in arr_o[:, 1]]) * self.f_d
        arr_T += np.divide(
            np.array(arr_t[:, 1]), arr_f_allc, out=np.zeros_like(arr_f_allc), where=np.array(arr_t[:, 1]) != 0
        )
        return arr_M, arr_T

