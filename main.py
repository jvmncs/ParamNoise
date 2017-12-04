from __future__ import print_function

import argparse
import os
import time
import random
import itertools

import gym

import torch
import torch.nn as nn
import torch.optim as optim

from core.models import DQN, PPO
from core.train import trainDQN, trainPPO
from core.test import testDQN, testPPO
from utils.utils import save_checkpoint, mkdir_p, AverageMeter, Logger
from utils.env import make_atari, wrap_deepmind, wrap_torch
from utils.storage import RolloutStorage, ReplayBuffer
from utils.progress.progress.bar import Bar

parser = argparse.ArgumentParser(description='Reproducing Parametric Noise')

# Setting up MDP
parser.add_argument('--env-id', type=str, metavar='ENV_ID',
                    help='which environment to train on')
parser.add_argument('--discount-factor', default=0.95, type=float, metavar='FLOAT',
                    help='reward discount factor')

# Noise-specific stuff
parser.add_argument('--alg', default='dqn', type=str, metavar='ALG',
                    choices = ['dqn', 'ppo'],
                    help='which algorithm to use')
parser.add_argument('--noise', default=None, metavar='NOISE_TYPE',
                    choices = [None, 'learned','adaptive'],
                    help='type of parameter noise to use')
# TODO: Incorporate these
parser.add_argument('--epsilon-greed', default = 1., type=float, metavar='FLOAT',
                    help='beginning of epsilon schedule (or constant)')
parser.add_argument('--epsilon-greed-end', default = .1, type=float, metavar='FLOAT',
                    help='end of epsilon schedule (meaningless if constant epsilon-greed)')
parser.add_argument('--epsilon-greed-steps', default = 1000000, type=int, metavar='N',
                    help='number of timesteps to linearly anneal over (must be equal to n-frames if constant)')
parser.add_argument('--noise-args', default = {}, type=dict, metavar='NOISE_ARGS',
                    help='arguments for the noise layers')

# DQN hyperparameters
parser.add_argument('--replay-memory', default=1000000, type=int, metavar='N',
                    help='replay memory size')
parser.add_argument('--memory-warmup', default=50000, type=int, metavar='N',
                    help='how many frames to warmup the replay memory (adaptive noise only)')
parser.add_argument('--sync-every', default=10000, type=int, metavar='N',
                    help='sync target net every n frames')

# PPO hyperparameters
parser.add_argument('--horizon', default=2048, type=int, metavar='T',
                    help='number of timesteps to generate experience for optimization')
parser.add_argument('--epochs', default=10, type=int, metavar='K',
                    help='number of epochs to optimize from experience')
parser.add_argument('--clip-epsilon', default=0.2, type=float, metavar='FLOAT',
                    help='clipping parameter in PPO surrogate loss')

# Optimization hyperparameters
parser.add_argument('--start-frame', default=0, type=int, metavar='N',
                    help='manual frame number (use in conjunction with -r)')
parser.add_argument('--batch-size', default=32, type=int, metavar='M',
                    help='train minibatch size')
parser.add_argument('--n-frames', default=10000000, type=int, metavar='N',
                    help='number of training frames')

# Miscellaneous
parser.add_argument('--eval-every', default=1000000, type=int, metavar='N',
                    help='evaluate every n frames')
parser.add_argument('--eval-period', default=500000, type=int, metavar='N',
                    help='number of frames per evaluation')
parser.add_argument('--max-episode-length', default=10000, type=int, metavar='N',
                    help='max frames per episode')
parser.add_argument('--manual-seed', default=None, type=int, metavar='N',
                    help='manual seed')
parser.add_argument('--disable-cuda', default=False, type=bool, metavar='BOOL',
                    help='disable cuda or nah')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()


# Duh
args.use_cuda = torch.cuda.is_available() and not args.disable_cuda


# Set Tensor types
args.FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
args.LongTensor = torch.cuda.LongTensor if args.use_cuda else torch.LongTensor
args.ByteTensor = torch.cuda.ByteTensor if args.use_cuda else torch.ByteTensor
args.Tensor = args.FloatTensor


# Set seeds
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manual_seed)


# Build env
env = gym.make(args.env_id)
is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
if is_atari:
    env = make_atari(env)
    env.seed(args.manual_seed)
    env = wrap_deepmind(env, frame_stack = True)
    # maybe figure out how to use this wrapper with frame_stack=True in wrap_deepmind
    # Until then, manually transpose states before they enter the net.
    # env = wrap_torch(env)
else:
    # TODO: Handle Mujoco wrapping as needed
    pass

# Sanity check
assert (args.alg == 'dqn' and is_atari) or (args.alg == 'ppo' and issubclass(env.action_space, gym.spaces.Box)), 'Algorithm must match up with environment.'


# Main training loop
def main(env, args):
    # Initiate args useful for training
    start_episode = 0
    args.current_frame = 0
    args.eval_start = 0
    args.test_num = 0
    args.test_time = False
    args.best_avg_return = -1

    # Make checkpoint path if there is none
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Instantiate metric tracking for progress bar
    # TODO: Add any other metrics we need to track
    # TODO: Still need to keep track of time for progress bar
    args.rewards  = AverageMeter()
    args.returns = AverageMeter()
    args.episode_lengths = AverageMeter()
    args.losses = AverageMeter()

    # Model & experiences
    print("==> creating model '{}' with '{}' noise".format(args.alg, args.noise))
    if args.alg == 'dqn':
        model = DQN(action_space = env.action_space, noise = args.noise)
        target_model = DQN(action_space = env.action_space, noise = args.noise)
        target_model.load_state_dict(model.state_dict())
        # Never going to train the target model
        target_model.eval()
        args.memory = ReplayBuffer(args.replay_memory, args.use_cuda)
    else:
        model = PPO(action_space = env.action_space, noise = args.noise, clip_epsilon = args.clip_epsilon)
        # TODO: Instantiate RolloutStorage
        # rollouts = RolloutStorage(args.horizon, arg.processes?,...)

    # House models on GPU if needed
    if args.use_cuda:
        model.cuda()
        if args.alg =='dqn':
            target_model.cuda()

    # Criterions and optimizers
    value_criterion = nn.functional.mse_loss
    if args.alg == 'dqn':
        if args.noise == 'adaptive':
            optimizer = optim.Adam(model.parameters(), lr = 1e-4)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr = 2.5e-4, momentum = 0.95, alpha = 0.95, eps = 1e-2)
    else:
        policy_criterion = model.surrogate_loss
        # TODO revisit the choices here.  Might be best to just go with defaults from PPO paper
        if args.noise == 'learned':
            optimizer = optim.RMSprop(model.parameters(), lr = 2.5e-4, momentum = 0.95, alpha = 0.95, eps = 1e-2)
        else:
            optimizer = optim.Adam(model.parameters(), lr = 3e-4)

    # Resume
    # Unload status, meters, and previous state_dicts from checkpoint
    print("==> resuming from '{}' at frame {}".format(args.resume, args.start_frame) if args.resume else "==> starting from scratch at frame {}".format(args.start_frame))
    title = str(args.noise) + '-' + args.env_id
    if args.resume:
        # Load checkpoint.
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_episode = checkpoint['episode'] + 1
        args.current_frame = checkpoint['frame'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        if args.alg == 'dqn':
            target_model.load_state_dict(checkpoint['target_state_dict'])
        args.returns = checkpoint['returns']
        args.best_avg_return = checkpoint['best_avg_return']
        args.episode_lengths = checkpoint['episode_lengths']
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.logger = Logger(os.path.join(args.checkpoint, title+'-log.txt'), title=title, resume=True)
    else:
        args.logger = Logger(os.path.join(args.checkpoint, title+'-log.txt'), title=title)
        args.logger.set_names(['Episode', 'Frame', 'EpLen', 'AvgLoss', 'Return'])

    # We need at least one experience in the replay buffer for DQN
    if args.alg == 'dqn':
        print("==> filling replay buffer with {} transition(s)".format(args.memory_warmup))
        state = env.reset()
        for i in range(args.memory_warmup):
            action = random.randrange(env.action_space.n)
            successor, reward, done, _ = env.step(action)
            args.memory.add(state, action, reward, successor, done)
            state = successor if not done else env.reset()
        # Need next reset to be a true reset (due to EpisodicLifeEnv)
        env.was_real_done = True

    print("==> beginning training")
    for episode in itertools.count(start_episode):
        # Train model
        if args.alg == 'dqn':
            env, model, target_model, optimizer, args = trainDQN(env, model, target_model, optimizer, value_criterion, args)
        else:
            env, model, optimizer, args = trainPPO(env, model, optimizer, value_criterion, policy_criterion, args)


        # Checkpoint model to disk
        is_best = args.returns.avg > args.best_avg_return
        if is_best:
            args.best_avg_return = args.returns.avg
        save_checkpoint({
            'episode': episode,
            'frame': args.current_frame,
            'state_dict': model.state_dict(),
            'target_state_dict': target_model.state_dict() if args.alg == 'dqn' else None,
            'rewards': args.rewards,
            'returns': args.returns,
            'best_avg_return': args.best_avg_return,
            'episode_lengths': args.episode_lengths,
            'losses': args.losses,
            'optimizer': optimizer.state_dict()
        }, is_best, title)

        # Log metrics (episode, frame, episode length, average loss, return)
        args.logger.append([episode, args.current_frame, args.episode_lengths.val, args.losses.avg, args.returns.val])

        # Reset frame-level meters
        args.losses.reset()
        args.rewards.reset()

        # Handle testing
        if args.test_time:
            # For testing only
            print("==> evaluating agent for {} frames at frame {}".format(args.eval_period, args.current_frame))

            args.eval_start = args.current_frame
            args.testing_frame = args.current_frame

            args.test_rewards  = AverageMeter()
            args.test_returns = AverageMeter()
            args.test_episode_lengths = AverageMeter()

            args.test_logger = Logger(os.path.join(args.checkpoint, 'test'+str(args.test_num)+'-'+title+'-log.txt'), title=title)
            args.test_logger.set_names(['Frame', 'EpLen', 'Return'])

            # Main testing loop
            while args.testing_frame - args.eval_start < args.eval_period:
                if args.alg == 'dqn':
                    env, args = testDQN(env, model, args)
                else:
                    env, args = testPPO(env, model, args)

                args.test_logger.append([args.testing_frame - args.eval_start, args.test_episode_lengths.val, args.test_returns.val])

                args.test_rewards.reset()

                # For testing only:
                #break
            # Need next reset to be a true reset
            env.was_real_done = True
            # Need to turn off testing for next episode
            args.test_time = False
            args.test_num += 1


        if args.current_frame > args.n_frames:
            break
        # For testing only:
        # if episode >= 100:
        #     break
        print('episode: ' + str(episode))
    # TODO: Handle cleanup
    env.close()

if __name__ == '__main__':
    main(env, args)
