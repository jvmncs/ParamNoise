from torch.autograd import Variable
from utils.utils import select_action

def testDQN(env, model, args):
    model.eval()
    state = env.reset()
    done = False
    initial_frame = args.current_frame
    while not done:
        # Handle a step
        action = select_action(state, model, args)
        successor, reward, done, _ = env.step(action)

        # Update frame-level meters
        args.test_rewards.update(float(reward))

        # Move on
        state = successor
        args.testing_frame += 1

    # Update episode-level meters
    args.test_returns.update(args.test_rewards.sum)
    args.test_episode_lengths.update(args.testing_frame - initial_frame)

    return env, args

def testPPO(env, model, args):
    model.eval()
    return env, args
