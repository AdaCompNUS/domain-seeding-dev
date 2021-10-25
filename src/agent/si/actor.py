import os
import time
import numpy as np
import torch
from torch.optim import AdamW
from pathlib import Path
import Pyro4
import sys

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent
print(f'workspace root: {ws_root}')
sys.stdout.flush()
SAVE_PATH = ws_root / 'trained_models'
sys.path.append(str(ws_root))

from utils.functions import print_flush, log_flush, error_handler_with_log, explained_variance_score
from utils.variables import data_host, log_port, replay_port
from memory.labelled import LabelledMemory

from agent.si.base import BaseOfflineAgent
from env.fetch_gym import FetchPushEnv
from random_sim.domain_randomization import ObjectRandomizer
from models.exploration import ExplorationPolicy


class SIActor(BaseOfflineAgent):
    space_size = 65

    def __init__(self, env, actor_id=0,
                 num_workers=8, num_epochs=10,
                 dataset_size=3e5, num_objects=1,
                 log_interval=1, memory_save_interval=5,
                 cuda=True, seed=0,
                 render=True):
        try:
            self.actor_id = actor_id
            self.render = render
            self.num_objects = num_objects
            self.log_interval = log_interval
            self.memory_save_interval = memory_save_interval
            self.num_workers = num_workers
            self.num_epochs = num_epochs
            memory_size = dataset_size/10
            self.termination_step = dataset_size
            self.log_flag = 'exploration/ms_{}_seed_{}'.format(memory_size, seed)
            self.log_txt = open("actor_log_{}.txt".format(actor_id), "w")

            self.writer = None
            self.env = env
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)

            self.device = torch.device(
                "cuda" if cuda and torch.cuda.is_available() else "cpu")
            log_flush(self.log_txt, 'device {}'.format(self.device))

            self.train_memory = LabelledMemory(
                memory_size, self.env.observation_space.shape,
                self.env.param_space.shape, self.device)

            Pyro4.config.COMMTIMEOUT = 0.0  # infinite wait
            Pyro4.config.SERIALIZER = 'pickle'
            self.logging_service = Pyro4.Proxy('PYRO:logservice.warehouse@{}:{}'.format(data_host, log_port))

            Pyro4.config.SERIALIZER = 'pickle'
            log_flush(self.log_txt, '[actor.py] ' + 'Connecting to replay service at '
                                          'PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
            self.replay_service = Pyro4.Proxy('PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
            # self.replay_service._pyroAsync()
            self.last_replay_block = 0

            self.steps = 0
            self.episodes = 0

            self.sim_randomizer = ObjectRandomizer()
            self.policy = ExplorationPolicy()

        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def __del__(self):
        self.log_txt.close()

    def run(self):
        try:
            self.time = time.time()
            while self.total_data_count() <= self.termination_step:
                self.episodes += 1
                self.act_episode()
                self.interval()
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def act_episode(self):
        try:
            log_flush(self.log_txt, "[actor.py] act_episode")
            episode_steps = 0
            episode_done = False

            prm_types, prm_argss = self.sim_randomizer.sample(num_objects=self.num_objects)
            obs, info = self.env.reset(prm_types=prm_types, prm_argss=prm_argss, mode='quick')
            obj_state = info['obj_state']

            if obs is None:
                log_flush(self.log_txt, 'Environment reset failed. Wasting episode')
                return

            if self.render:
                self.env.render()

            obs_seq, reward, succeed, info1 = self.env.step(self.policy.next_action(info['obj_state']))
            self.steps += 1
            if succeed:
                self.update_memory(obs_seq, obj_state.serialize())
                print_flush(f'[actor.py ]Episode succeed, recording data with {obs_seq.shape} obs channels')
            else:
                print_flush('[actor.py ]Episode failed, not recording data')

            now = time.time()
            print(' ' * self.space_size,
                  f'Actor {self.actor_id:<2}  '
                  f'episode: {self.episodes:<4}  '
                  f'episode steps: {episode_steps:<4}  '
                  f'time: {now - self.time:3.3f}')
            self.time = now

            if self.episodes % self.log_interval == 0:
                log_dict = {
                    'actor_alive/{}'.format(self.actor_id): True
                }
                self.logging_service.add_log(1, self.log_flag, log_dict)

        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def interval(self):
        try:
            if self.episodes % self.memory_save_interval == 0:
                self.save_memory(self.actor_id)
        except Exception as e:
            error_handler_with_log(self.log_txt, e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid',
                        type=int,
                        default=0,
                        help='actor_id')
    parser.add_argument('--port',
                        type=int,
                        default=2000,
                        help='summit_port')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    use_data = False
    with FetchPushEnv(gui=False) as fetch_env:
        config = {
            'env': fetch_env,
            'cuda': True,
            'render': True,
            'actor_id': args.aid,
            'seed': args.seed,
            'num_objects': 1,
            'dataset_size': 120000,
            'log_interval': 10,
            'memory_save_interval': 5,
        }

        actor = SIActor(**config)
        actor.run()

    print_flush('[actor.py] termination')
