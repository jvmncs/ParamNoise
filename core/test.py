from torch.autograd import Variable
from utils.utils import select_action

def testDQN(env, model, args):
    # Initiate stuff
    model.eval()
    state = env.reset() # True reset
    done = False
    initial_frame = args.testing_frame
    # Eval for one episode (i.e. one life)
    while not done:
        # Handle a step
        action = select_action(state, model, args)
        successor, reward, done, _ = env.step(action)

        # Update frame-level meters
        args.test_rewards.update(float(reward))

        # Move on
        state = successor
        args.testing_frame += 1

        # Update testing progress bar
        args.test_bar.suffix = '({frame}/{size}) | Total: {total:} | ETA: {eta:} | AvgReward: {rewards: .4f}'.format(
                    frame=args.testing_frame - args.eval_start,
                    size=args.eval_period,
                    #data=data_time.avg,
                    #bt=batch_time.avg,
                    total=args.bar.elapsed_td,
                    eta=args.bar.eta_td,
                    rewards=args.test_rewards.avg)
        args.test_bar.next()

        if args.testing_frame - initial_frame >= args.max_episode_length:
            break

    # Update episode-level meters
    args.test_returns.update(args.test_rewards.sum)
    args.test_episode_lengths.update(args.testing_frame - initial_frame)

    args.test_bar.suffix += ' | Return {return_} | Episode Length {length}\n'.format(
                return_=args.test_returns.val,
                length=args.test_episode_lengths.val)
    args.test_bar.next()

    return env, args

def testPPO(env, model, args):
    model.eval()
    return env, args
