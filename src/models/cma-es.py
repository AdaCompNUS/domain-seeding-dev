import sys
import os
import cma
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Type, List

ws_root = Path(os.path.realpath(__file__)).parent.parent
print(f'workspace root: {ws_root}')
sys.stdout.flush()
sys.path.append(str(ws_root))

from utils.variables import PARAM_SEARCH_BOUNDS
from utils.classes import ParameterizedPolicy
from env.fetch_gym import FetchPushEnv
from models.exploration import ExplorationPolicy


def max_range():
    ret = 0.0
    for bound in PARAM_SEARCH_BOUNDS:
        ret = max(ret, bound[1] - bound[0])
    return ret


class CMA_ES:
    def __init__(self, env: FetchPushEnv, policy_class: Type, init_x: List[float], pop_size: int, end_iter=100):
        self.env = env
        self.policy_class = policy_class
        self.init_x = init_x
        self.pop_size = pop_size
        self.dim_x = policy_class.dim
        self.end_iter = end_iter

    def objective_func(self, x):
        self.env.reset(reset_env=True, reset_robot=True, reset_goal=False)
        _, reward, succeed, _ = self.env.step(self.policy_class(x), mode='super')
        print('reward, succeed: {} {}'.format(reward, succeed))
        sys.stdout.flush()
        return reward

    def search(self):
        # search for the best policy parameter
        es = cma.CMAEvolutionStrategy(self.init_x, 0.25 * max_range(), {'popsize': self.pop_size})
        i = 0
        while not es.stop() and i < self.end_iter:
            (X, fitness) = es.ask_and_eval(self.objective_func)
            es.tell(X, fitness)
            es.logger.add()  # write data to disc to be plotted
            es.disp()
            # es.result_pretty()
            i += 1
        res = es.result()
        print('search result: {}'.format(res))

        cma.plot()  # shortcut for es.logger.plot()
        cma.s.figsave('execution.png')
        input("Press Enter to continue.")
        return res


def f(x):
    # return numpy.linalg.norm(x*2-1/x+ numpy.exp(x))
    return cma.ff.rosen(x)


if __name__ == '__main__':
    from random_sim.domain_randomization import ObjectRandomizer
    from utils.classes import TaskGoal, GType

    env = FetchPushEnv(gui=True)
    env.render()
    randomizer = ObjectRandomizer()
    prm_types, prm_argss = randomizer.sample(num_objects=1)
    _, _ = env.reset(prm_types=prm_types, prm_argss=prm_argss, goal=TaskGoal(GType.BEYOND, [0, 1.0, 0.8]))

    init_policy = ExplorationPolicy()
    policy = CMA_ES(env=env, policy_class=ParameterizedPolicy, pop_size=10, end_iter=10,
                    init_x=init_policy.next_action(None).serialize())
    ret = policy.search()

    input()
