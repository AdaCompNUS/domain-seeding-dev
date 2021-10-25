import Pyro4
import torch
import time
from decimal import Decimal
from pathlib import Path
import os
import sys

ws_root = Path(os.path.realpath(__file__)).parent.parent
sys.path.append(str(ws_root))

from utils.functions import print_flush, error_handler
from agent.utils import to_batch


class BaseOfflineAgent:

    def __init__(self):
        self.env = None
        self.device = None
        self.shared_memory = None
        self.shared_weights = dict()
        self.train_memory = None
        self.test_memory = None
        self.replay_service = None
        self.last_replay_block = None

    def run(self):
        raise Exception('You need to implement run method.')

    def interval(self):
        raise Exception('You need to implement interval method.')

    def update_params(self, optim, network, loss, grad_clip=None):
        try:
            mean_grads = None
            optim.zero_grad()
            loss.backward(retain_graph=True)
            if grad_clip is not None:
                for p in network.modules():
                    torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
            if network is not None:
                mean_grads = self.calc_mean_grads(network)
            optim.step()
            return mean_grads
        except Exception as e:
            error_handler(e)

    def calc_mean_grads(self, network):
        total_grads = 0
        for m in network.modules():
            for p in m.parameters():
                total_grads += p.grad.clone().detach().sum()
        return total_grads / network.num_params

    def calc_grad_mag(self, network):
        n = 0
        grad_norm = 0
        for m in network.modules():
            for p in m.parameters():
                if p.grad is not None:
                    grad_norm = (n * grad_norm + p.grad.clone().detach().norm()) / (n + 1)
                    n = n + 1
        return grad_norm

    def total_data_count(self):
        try:
            return self.replay_service.total_data_count()
        except Exception as e:
            error_handler(e)

    def load_memory(self, is_train=True, max_blocks=10):
        try:
            self.last_replay_block, new_memory = self.replay_service.fetch_memory(self.last_replay_block)
            parsed_blocks = 0
            while new_memory is not None and parsed_blocks < max_blocks:
                print_flush('[leaner_base] loaded memory at block {}'.format(self.last_replay_block))
                if is_train:
                    self.train_memory.load(new_memory)
                else:
                    self.test_memory.load(new_memory)
                print_flush('[leaner_base] memory size: train {}, val {}'.format(
                    len(self.train_memory), len(self.test_memory)))
                self.last_replay_block, new_memory = self.replay_service.fetch_memory(self.last_replay_block)
                parsed_blocks += 1
        except Exception as e:
            error_handler(e)

    def save_memory(self, actor_id):
        print_flush("[actor_base] save_memory")
        if len(self.train_memory) > 1:
            key = '{:.9f}'.format(Decimal(time.time()))
            self.replay_service.add_memory(key, self.train_memory.get(), actor_id)
        self.train_memory.reset()

    def update_memory(self, images, label):
        try:
            assert(images is not None)
            assert(label is not None)
            self.train_memory.append(images, label)
        except Exception as e:
            error_handler(e)

