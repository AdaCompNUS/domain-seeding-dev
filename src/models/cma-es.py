import sys
import os
import cma
import math
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Type, List

ws_root = Path(os.path.realpath(__file__)).parent.parent
print(f'workspace root: {ws_root}')
sys.stdout.flush()
sys.path.append(str(ws_root))

from utils.variables import LOGGING_INFO, LOGGING_MIN, LOGGING_DEBUG
from utils.classes import ParameterizedPolicy
from env.fetch_gym import FetchPushEnv
from models.exploration import ExplorationPolicy
from random_sim.domain_randomization import PrimitiveRandomizer


trial = 0
succeed = 0


def max_range():
    ret = 0.0
    for (min_v, max_v) in zip(PrimitiveRandomizer.SEARCH_LB, PrimitiveRandomizer.SEARCH_UB):
        ret = max(ret, max_v - min_v)
    return ret


class CMA_ES:
    def __init__(self, env: FetchPushEnv, policy_class: Type, init_x: List[float], pop_size: int, end_iter=100,
                 logging_level=LOGGING_MIN):
        self.env = env
        self.policy_class = policy_class
        self.init_x = init_x
        self.pop_size = pop_size
        self.end_iter = end_iter
        self.logging_level = logging_level
        env.logging_level = logging_level

    def print_with_level(self, msg, level=LOGGING_DEBUG):
        if self.logging_level >= level:
            print(msg)

    def objective_func(self, x):
        global trial, succeed
        self.env.reset(reset_env=False, reset_object=True, reset_robot=True, reset_goal=False, need_return=False,
                       mode='super')
        self.print_with_level(f'Search action {x}', LOGGING_INFO)
        _, reward, succ, _ = self.env.step(self.policy_class(x), generate_obs=False)
        # env.render()
        if not succ:
            self.print_with_level('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`reward, succeed: {} {}'.format(reward, succ),
                                  LOGGING_DEBUG)
        sys.stdout.flush()
        trial += 1
        succeed += succ
        return -reward

    def search(self):
        # search for the best policy parameter
        global trial, succeed
        # es = cma.CMAEvolutionStrategy(self.init_x, 0.1 * max_range(), {'popsize': self.pop_size})
        es = cma.CMAEvolutionStrategy(self.init_x, 0.2, {'popsize': self.pop_size})

        i = 0
        while not es.stop() and i < self.end_iter:
            trial, succeed = 0, 0
            (X, fitness) = es.ask_and_eval(self.objective_func)
            es.tell(X, fitness)
            es.logger.add()  # write data to disc to be plotted
            es.disp()
            # es.result_pretty()
            self.print_with_level(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`succeeded in {succeed} / {trial} trials',
                                  LOGGING_MIN)
            i += 1

        res = es.result.xfavorite
        self.print_with_level('search result: {}'.format(es.result), LOGGING_MIN)

        cma.plot()  # shortcut for es.logger.plot()
        cma.s.figsave('execution.png')
        self.print_with_level("Press Enter to continue.", LOGGING_MIN)
        return res


def f(x):
    # return numpy.linalg.norm(x*2-1/x+ numpy.exp(x))
    return cma.ff.rosen(x)


if __name__ == '__main__':
    from random_sim.domain_randomization import ObjectRandomizer
    from utils.classes import TaskGoal, GType

    env = FetchPushEnv(gui=False, logging_level=LOGGING_MIN)
    # env.render()
    randomizer = ObjectRandomizer()
    prm_types, prm_argss = randomizer.sample(num_objects=1)
    obs, info = env.reset(prm_types=prm_types, prm_argss=prm_argss,
                          # goal=TaskGoal(GType.BEYOND_WITHIN, [0.0, 1.0, 0.7, 0.0, 1.0, 1.0]))
                          goal=TaskGoal(GType.WITHIN_CIRCLE, [0.3, 0.6, 0.2]))

    # input()
    # env.logging_level = LOGGING_INFO

    init_policy = ExplorationPolicy()
    env.step(action=init_policy.next_action(info['obj_state']))
    # input()

    print(f"Initial action {init_policy.next_action(info['obj_state']).serialize()}")
    policy = CMA_ES(env=env, policy_class=ParameterizedPolicy, pop_size=50, end_iter=50,
                    init_x=init_policy.next_action(info['obj_state']).serialize(), logging_level=LOGGING_MIN)
    ret = policy.search()
    # input()

    print('Resetting env')
    env.logging_level = LOGGING_DEBUG
    # input()
    while True:
        _, _ = env.reset(reset_env=False, reset_object=True, reset_robot=True, reset_goal=False, mode='normal')
        print('Conducting best policy')
        env.step(action=ParameterizedPolicy(x=ret))
        input()

