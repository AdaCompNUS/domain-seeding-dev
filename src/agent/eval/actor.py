import os
import sys
from pathlib2 import Path
import torch


ws_root = Path(os.path.realpath(__file__)).parent.parent.parent
print(f'workspace root: {ws_root}')
sys.stdout.flush()
sys.path.append(str(ws_root))

from models.si_model import ISModel
from models.exploration import ExplorationPolicy
from models.cmaes import CMA_ES
from utils.classes import ParameterizedPolicy, ObjState, PType, PPos, POrientation, PScale, PCom, PPhysics
from utils.variables import LOGGING_MIN, LOGGING_INFO, LOGGING_DEBUG
from utils.functions import print_flush
from simulation.domain_randomization import PrimitiveRandomizer
from env.fetch_gym import FetchPushEnv


LOAD_PATH = ws_root / 'trained_models'


class EvalActor:
    def __init__(self, env, cuda=True, si_model_name="si.pth", logging_level=LOGGING_MIN,
                 seed=None, aid=0, render=True):
        self.real_env = env
        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.logging_level = logging_level

        self.exp_pi = ExplorationPolicy()
        self.si = self.model = ISModel(
                self.real_env.observation_space.shape[0],
                self.real_env.param_space.shape[0]).to(self.device)
        self.si.load_state_dict(torch.load(str(LOAD_PATH / si_model_name)))
        self.si.eval()

    def print_with_level(self, msg, level=LOGGING_DEBUG):
        if self.logging_level >= level:
            print(msg)

    def run(self):
        """Run the exploration policy in the real env,
        get the observation sequence"""
        _, info = self.real_env.reset(reset_env=False, reset_object=True, reset_robot=True,
                                   reset_goal=False, mode='quick')
        obj_state = info['obj_state']
        obs_seq, reward, succeed, info1 = self.real_env.step(self.exp_pi.next_action(obj_state))

        """Query the system identification model to guess the model parameters"""
        inputs = torch.FloatTensor(obs_seq).unsqueeze(0)
        param_pred = PrimitiveRandomizer.mean_cemas_params(obj_state.type)
        self.logging_level(f'Start from param {param_pred}', LOGGING_MIN)
        param_diff = self.si(inputs, param_pred).cpu().detach().squeeze(0)
        while ObjState.norm(param_diff) > 1e-2:
            # the model gives directions to update the reference
            self.logging_level(f'Applying param diff {param_diff}', LOGGING_MIN)
            param_pred = [ref_x + diff_x for (ref_x, diff_x) in zip(param_pred, param_diff)]
            param_diff = self.si(inputs, param_pred).cpu().detach().squeeze(0)

        """Reset the environment to initial state
        In reality, this can be different from the previous starting state"""
        _, info = self.real_env.reset(reset_env=False, reset_object=True, reset_robot=True,
                                      reset_goal=False, mode='quick')
        obj_state = info['obj_state']

        """Construct a new simulation according to the model parameters"""
        sim_env = FetchPushEnv(gui=True)
        p_args = {}
        p_type = obj_state.type
        p_args['pos'] = PPos(obj_state.pos, p_type)
        p_args['rot'] = POrientation.from_quaternion(obj_state.quaternion, p_type)
        p_args['shape'], p_args['com'], p_args['phy'] = PrimitiveRandomizer.refract_cemas_params(param_pred, p_type)
        obs, info = sim_env.reset(prm_types=[p_type], prm_argss=[p_args], mode='quick')

        """Policy search using CMA-ES"""
        init_policy = ExplorationPolicy()
        policy = CMA_ES(env=sim_env, policy_class=ParameterizedPolicy, pop_size=50, end_iter=50,
                        init_x=init_policy.next_action(info['obj_state']).to_cmaes(), logging_level=LOGGING_MIN)

        ret = policy.search()

        """Policy execution in the actual environment"""
        print('Resetting env')
        self.real_env.logging_level = LOGGING_DEBUG
        # input()
        while True:
            _, _ = self.real_env.reset(reset_env=False, reset_object=True, reset_robot=True, reset_goal=False, mode='normal')
            print('Conducting best policy')
            self.real_env.step(action=ParameterizedPolicy(x=ret))
            input()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--aid',
                        type=int,
                        default=0,
                        help='actor_id')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    use_data = False
    with FetchPushEnv(gui=True) as fetch_env:
        config = {
            'env': fetch_env,
            'cuda': True,
            'render': True,
            'actor_id': args.aid,
            'seed': args.seed,
            'si_model_name': "si.pth",
            'logging_level': LOGGING_MIN
        }

        actor = EvalActor(**config)
        actor.run()

    print_flush('[actor.py] termination')
