from torch.autograd import Variable
from utils.utils import select_action

# Only one episode (Remember, end of life = end of episode for DQN)
def trainDQN(env, model, target_model, optimizer, value_criterion, args):
    model.train()
    state = env.reset()
    done = False
    initial_frame = args.current_frame
    # Handle training for one episode
    while not done:
        # Sample noise if needed:
        if args.noise == 'learned':
            model.reset_noise()

        # Handle training for one step
        action = select_action(state, model, args)
        successor, reward, done, _ = env.step(action)
        args.memory.add(state, action, reward, successor, done)

        # Sample from replay buffer and prepare
        states, actions, rewards, successors, dones = args.memory.sample(args.batch_size)
        final_mask = args.ByteTensor(tuple(map(lambda s: s is None, successors)))
        states = Variable(states)
        actions = Variable(actions)
        rewards = Variable(rewards)

        # Resample noise if needed
        if args.noise == 'learned':
            model.reset_noise()
            target_model.reset_noise()

        # Compute terms for network update
        Q_values = model(states).max(1)[0]
        states.volatile = True
        target_Q_values = target_model(states).max(1)[0]
        target_Q_values[final_mask] = 0
        target_Q_values.volatile = False
        expected_Q_values = args.discount_factor * target_Q_values + rewards

        # Compute loss, backpropagate and apply clipped gradient update
        loss = value_criterion(Q_values, target_Q_values)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        # Sync target network if needed
        if args.current_frame % args.sync_every == args.sync_every - 1:
            target_model.load_state_dict(model.state_dict())

        # Update frame-level meters
        args.losses.update(loss.data[0])
        args.rewards.update(float(reward))

        # Move on
        state = successor
        args.current_frame += 1

    # Update episode-level meters
    args.returns.update(args.rewards.sum)
    args.episode_lengths.update(int(args.current_frame - initial_frame))

    # Initiate evaluation if needed
    if args.current_frame - args.eval_start > args.eval_every:
        args.test_time = True

    return env, model, target_model, optimizer, args


def trainPPO(env, model, optimizer, value_criterion, policy_criterion, args):
    model.train()
    state = env.reset()
    return env, model, optimizer, args
