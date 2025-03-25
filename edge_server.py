import numpy as np
from agent import Edge_Agent
from agent import Edge_Agent_Risk


# noinspection PyUnresolvedReferences, NonAsciiCharacters, PyTypeChecker
class Edge_Server:
    DEFAULTS = {
        "D": [200, 400, 600, 800],  # Computation demand (10^12 CPU cycles)
        "A_u": [0, 100, 200, 300, 400, 500, 600],  # Request amount of universal resource (10^12 CPU cycles)
        "A_d": [0, 100, 200, 300, 400],  # Purchase amount of dynamic resource (10^12 CPU cycles)
        "f_u": 100,  # Hourly frequency of cloud universal resource (10^12 CPU cycles/h)
        "f_d": 100,  # Hourly frequency of cloud dynamic resource (10^12 CPU cycles/h)
        "f_l": 50,  # Hourly frequency of local resource (10^12CPU cycles/h)
        "ρ_l": 3.0,  # Hourly price of edge's local resource (RMB)
        "p_max": 400,  # Maximum amount of excess purchased universal resource
        "t_avg": 0.15,  # Average CPU cycles of tasks (10^12 CPU cycles)
        "l": 0.04  # Average transmission latency of tasks (s)
    }

    def __init__(self, config):
        self.__dict__.update(Edge_Server.DEFAULTS, **config)

        self.rng = np.random.default_rng(self.seed)
        # resource demand from users
        self.d = 0
        # obtained cloud universal / dynamic resources after purchasing
        self.arr_o = np.zeros(2)
        # task schedule for universal / dynamic / local resources
        self.arr_t = np.zeros(3)
        # usage cost / avg task time
        self.M, self.T = 0, 0

        # setup agent for generating optimum resource purchase strategy
        self.agent = object
        self.setup()

    def setup(self):
        # setup agent
        config = {
            "dims": 3,
            "dima": 2,
            "γ": self.γ,
            "σ": self.σ,
            "capacity": self.capacity,
            "batch": self.batch
        }
        # security scenario with information keeps private
        if self.is_security:
            self.agent = Edge_Agent(config)
        else:
            config['dims'] = 4
            self.agent = Edge_Agent_Risk(config)

    def env_change(self):
        self.d = self.rng.choice(self.D)

    def update_state(self, n1, n2, n3=None):
        # secure state : ρ_c , d, p
        if n3 is None:
            self.agent.update_state({
                "ρ_c": n1,
                "d": self.d,
                "p": n2
            })
        # insecure state: q, d, p, d-
        else:
            self.agent.update_state({
                "q": n1,
                "d": self.d,
                "p": n2,
                "d-": n3
            })

    def resource_purchase(self, policy='Rand'):
        """
        Make resource purchase decisions a_u, a_d
        return:
        a_u: purchased universal resources
        a_d: purchased dynamic resources
        """
        match policy:
            case 'Rand':
                a_u, a_d = self.rng.choice(self.A_u), self.rng.choice(self.A_d)
            case 'RL':
                a_u, a_d = self.agent.generate_action()
                if a_u < 0:
                    a_u, a_d = self.rng.choice(self.A_u), self.rng.choice(self.A_d)
            case _:
                a_u, a_d = self.rng.choice(self.A_u), self.rng.choice(self.A_d)
        return a_u, a_d

    def task_schedule(self, o_u, o_d):
        """
        Schedule tasks for execution based on obtained resources o_u / o_d
        return:
        t_u: universal resources for task execution
        t_d: dynamic resources for task execution
        p:   excess purchased universal resources
        """
        # previous excess purchased universal resources
        pre_p = self.agent.get_state("p")

        #
        if self.d <= o_d:
            t_u, t_d = 0, self.d
            p = min(o_u + pre_p, self.p_max)
        #
        else:
            t_d = o_d
            t_u = min(self.d - o_d, o_u + pre_p)
            p = pre_p + o_u + o_d - self.d
            p = np.clip(p, 0, self.p_max)
        #
        self.arr_o = [o_u, o_d]
        self.arr_t = [t_u, t_d, self.d - t_u - t_d]
        return t_u, t_d, p

    def task_execution(self):
        """
        manipulate local task execution based on scheduled resources
        return:
        M: local usage cost
        T: local task time
        """
        #
        M, T = 0, 0
        # task for local resources execution
        if self.arr_t[2] > 0:
            T = self.arr_t[2] / self.f_l
            M = T * self.ρ_l
        self.M, self.T = M, T

    def reward_calculation(self, M_cloud, T_cloud):
        """
        :param M_cloud: cloud service expense
        :param T_cloud: cloud service time
        :return:

        """
        # overall usage cost = local cost + cloud cost
        self.M += M_cloud

        # overall task time = local execution time + cloud execution time + cloud transmission time
        T_tran = np.sum(self.arr_t[:2]) / self.t_avg * self.l  # (s)
        T_exec = (self.T + T_cloud) * 3600  # (s)
        # avg task time = overall task time / task nums
        self.T = np.round((T_tran + T_exec) / self.d * self.t_avg, 2)

        # reward calculation
        ρ_c = self.agent.get_state("ρ_c")
        server_time = self.arr_o[0] / self.f_u + self.arr_o[1] / self.f_d + self.arr_t[2] / self.f_l
        ρ_a = np.round(self.M / server_time if server_time != 0 else 0.0, 2)
        r_c = np.round((-np.abs(ρ_c - ρ_a)), 2)
        r_p = np.round((- self.ξ * self.M - (1 - self.ξ) * self.T), 2)
        return r_c, r_p
