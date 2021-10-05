import os
from time import time
import numpy as np
import torch
from torch.optim import AdamW
from datetime import datetime
from pathlib import Path
import Pyro4
import sys
import pickle

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent
print('workspace root: {ws_root}')
sys.stdout.flush()
SAVE_PATH = ws_root / 'trained_models'
sys.path.append(str(ws_root))

from utils.functions import print_flush, log_flush, error_handler_with_log, explained_variance_score
from utils.variables import data_host, log_port, replay_port
from utils.classes import AverageMeter
from models.si_model import ISModel
from memory.labelled import LabelledMemory

from agent.si.base import BaseOfflineAgent
from fetch_gym import FetchPushEnv


class SILearner(BaseOfflineAgent):

    def __init__(self, env, learner_id=0,
                 num_workers=8, num_epochs=10,
                 batch_size=64, lr=0.0003,
                 dataset_size=3e5, grad_clip=5.0,
                 log_interval=1, memory_load_interval=5,
                 model_checkpoint_interval=1,
                 model_save_interval=5,
                 cuda=True, seed=0):
        try:
            self.lr = lr
            self.batch_size = batch_size
            self.grad_clip = grad_clip
            self.log_interval = log_interval
            self.memory_load_interval = memory_load_interval
            self.model_checkpoint_interval = model_checkpoint_interval
            self.model_save_interval = model_save_interval
            self.learner_id = str(learner_id)
            self.num_workers = num_workers
            self.num_epochs = num_epochs
            memory_size = dataset_size
            self.start_steps = dataset_size - 1
            self.termination_step = dataset_size * 2
            self.log_flag = 'si/lr_{}_bs_{}_ms_{}_seed_{}'.format(lr, batch_size, memory_size, seed)
            self.log_txt = open("leaner_log_{}.txt".format(learner_id), "w")

            self.writer = None
            self.env = env
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            self.device = torch.device(
                "cuda" if cuda and torch.cuda.is_available() else "cpu")

            log_flush(self.log_txt, 'device {}'.format(self.device))

            torch.autograd.set_detect_anomaly(True)

            self.model = ISModel(
                self.env.observation_space.shape[0],
                self.env.param_space.n).to(self.device)

            self.loss = torch.nn.MSELoss()
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, min_lr=0.0001, factor=0.63, patience=10000)

            self.steps = 0
            self.save_weights(leaner_id=learner_id)

            self.memory = LabelledMemory(
                memory_size, self.env.observation_space.shape,
                self.env.param_space.n, self.device)

            self.model_dir = str(SAVE_PATH / self.learner_id)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

            Pyro4.config.COMMTIMEOUT = 0.0  # infinite wait
            Pyro4.config.SERIALIZER = 'pickle'
            self.logging_service = Pyro4.Proxy('PYRO:logservice.warehouse@{}:{}'.format(data_host, log_port))

            Pyro4.config.SERIALIZER = 'pickle'
            log_flush(self.log_txt, '[learner.py] ' + 'Connecting to replay service at '
                                          'PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
            self.replay_service = Pyro4.Proxy('PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
            # self.replay_service._pyroAsync()
            self.last_replay_block = 0

            self.epochs = 0
            self.train_loader = None
            self.validation_loader = None

        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def __del__(self):
        self.log_txt.close()

    def run(self):
        try:
            self.load_memory()

            self.time = time()
            while len(self.memory) < self.start_steps:
                self.load_memory()

            # self.time = time()
            self.train_loader = torch.utils.data.DataLoader(
                self.train_memory, batch_size=self.batch_size,
                num_workers=self.num_workers, pin_memory=True, shuffle=True)
            self.validation_loader = torch.utils.data.DataLoader(
                self.test_memory, batch_size=self.batch_size,
                num_workers=self.num_workers, pin_memory=True, shuffle=False)
            for epoch in range(self.num_epochs):
                self.epochs = epoch
                self.learn()
                self.evaluate()
                self.interval()
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def learn(self):
        try:
            log_flush(self.log_txt, 'Learning for epoch {}'.format(self.epochs))
            start_time = time.time()

            self.model.train()
            for i, (images, label) in enumerate(self.train_loader):
                self.steps += 1
                images, label = images.cuda(), label.cuda()
                prediction = self.model(images)
                loss = self.loss(prediction, label)
                self.update_params(
                    self.optimizer, self.model, loss, self.grad_clip)
                if self.steps % self.log_interval == 0:
                    prediction = prediction.detach()
                    error = torch.abs(label - prediction)
                    cur_lr = self.optimizer.param_groups[0]['lr']
                    evars = []
                    for i in range(self.env.param_space.n):
                        evars.append(explained_variance_score(label.cpu()[:, i], prediction.detach().cpu()[:, i]))
                    grad_norm = self.calc_grad_mag(self.model)

                    log_dict = {
                        'loss/mse': loss.detach().item(),
                        'loss/evar0': evars[0],
                        'loss/evar1': evars[1],
                        'loss/evar2': evars[2],
                        'stats/lr': cur_lr,
                        'stats/mean0': prediction[:, 0].mean().item(),
                        'stats/mean1': prediction[:, 1].mean().item(),
                        'stats/mean2': prediction[:, 2].mean().item(),
                        'stats/max_error0': error[:, 0].max().item(),
                        'stats/max_error1': error[:, 1].max().item(),
                        'stats/max_error2': error[:, 2].max().item(),
                        'stats/grad_norm': grad_norm,
                        'progress/epoch': self.epochs,
                    }
                    self.logging_service.add_log(self.log_interval, self.log_flag, log_dict)
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def interval(self):
        if self.epochs % self.model_save_interval == 0:
            self.save_weights(self.learner_id)
        if self.epochs % self.model_checkpoint_interval == 0:
            self.save_models()

    def evaluate(self):
        try:
            log_flush(self.log_txt, 'Evaluating for epoch {}'.format(self.epochs))
            self.model.eval()
            for i, (images, label) in enumerate(self.validation_loader):
                self.steps += 1
                images, label = images.cuda(), label.cuda()
                with torch.no_grad():
                    prediction = self.model(images)
                    loss = self.loss(prediction, label)

                    if self.steps % self.log_interval == 0:
                        prediction = prediction.detach()
                        error = torch.abs(label - prediction)
                        evars = []
                        for i in range(self.env.param_space.n):
                            evars.append(explained_variance_score(label.cpu()[:, i], prediction.detach().cpu()[:, i]))
                        grad_norm = self.calc_grad_mag(self.model)

                        log_dict = {
                            'loss/mse_val': loss.detach().item(),
                            'loss/evar0_val': evars[0],
                            'loss/evar1_val': evars[1],
                            'loss/evar2_val': evars[2],
                            'stats/mean0_val': prediction[:, 0].mean().item(),
                            'stats/mean1_val': prediction[:, 1].mean().item(),
                            'stats/mean2_val': prediction[:, 2].mean().item(),
                            'stats/max_error0_val': error[:, 0].max().item(),
                            'stats/max_error1_val': error[:, 1].max().item(),
                            'stats/max_error2_val': error[:, 2].max().item(),
                            'stats/grad_norm_val': grad_norm,
                            'progress/epoch_val': self.epochs,
                        }
                        self.logging_service.add_log(self.log_interval, self.log_flag, log_dict)
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def save_models(self):
        torch.save(self.model.state_dict(), str(SAVE_PATH / self.learner_id / 'is_model_cp_{}'.format(self.steps)))

    def save_weights(self, leaner_id):
        try:
            log_flush(self.log_txt, '[learner.py] save weights to {}'.format(str(SAVE_PATH / str(leaner_id))))
            leaner_id = str(leaner_id)
            if not os.path.isdir(str(SAVE_PATH / leaner_id)):
                os.makedirs(str(SAVE_PATH / leaner_id))

            torch.save(self.model.state_dict(), str(SAVE_PATH / leaner_id / 'policy'))
        except Exception as e:
            error_handler_with_log(self.log_txt, e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='GPU to use')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0003,
                        help='learning rate')
    parser.add_argument('--port',
                        type=int,
                        default=2000,
                        help='summit_port')
    parser.add_argument('--drive_mode',
                        type=str,
                        default="lets-drive-zero",
                        help='Which drive_mode to run')
    parser.add_argument('--env_mode',
                        type=str,
                        default="server",
                        help='display or server')
    parser.add_argument('--offline',
                        type=bool,
                        default=False,
                        help='offline training mode')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    use_data = False
    with FetchPushEnv(gui=False) as fetch_env:
        config = {
            'env': fetch_env,
            'cuda': True,
            'seed': args.seed,
            'batch_size': 64,
            'lr': args.lr,
            'num_workers': 8,
            'num_epochs': 30,
            'grad_clip': 5.0,
            'dataset_size': 3e5,
            'log_interval': 10,
            'memory_load_interval': 5,
            'model_checkpoint_interval': 1,
            'model_save_interval': 5
        }

        learner = SILearner(**config)
        learner.run()

    # env_server.send_signal(signal.SIGKILL)
    # env_server.wait(timeout=5)

    print_flush('[learner.py] termination')
