from torch.autograd import Variable
from utils.utils import select_action

# Only one episode (Remember, end of life = end of episode for DQN)
def trainDQN(env, model, target_model, optimizer, value_criterion, args):
    model.train()
    state = env.reset()
    done = False
    initial_frame = args.current_frame
    loss = 0
    # Handle training for one episode
    while not done:
        # Sample noise if needed:
        if args.noise == 'learned':
            model.resample()
        elif args.noise == 'adaptive':
            model.renoise()
            model.resample()

        # Take a step
        action = select_action(state, model, args)
        successor, reward, done, _ = env.step(action)
        args.memory.add(state, action, reward, successor, done)

        # Sample from replay buffer and prepare
        states, actions, rewards, successors, dones = args.memory.sample(args.batch_size)
        final_mask = args.ByteTensor(tuple(map(lambda s: s is None, successors)))
        states = Variable(states)
        actions = Variable(actions)
        rewards = Variable(rewards)

        # Handle noise stuff if needed
        if args.noise == 'adaptive':
            model.denoise()
            target_model.denoise()

        # Compute terms for network update
        if args.noise == 'learned':
            # Need to resample parameters for each transition in the batch
            # I wonder if this is really necessary
            # Seems inefficient at best but I don't make the rules
            for i in range(len(states.split(1))):
                model.resample()
                target_model.resample()
                this_one = states[i]
                Q = model(this_one).max(1)[0]
                this_one.volatile = True
                target_Q = target_model(this_one).max(1)[0] if not final_mask[i] else 0
                target_Q.volatile = False
                expected_Q = args.discount_factor * target_Q + rewards[i]

                # Compute loss, backpropagate and apply clipped gradient update
                one_step_loss = value_criterion(Q, expected_Q)
                optimizer.zero_grad()
                one_step_loss.backward()
                for param in model.parameters():
                    param.grad.data.clamp(-1,1)
                optimizer.step()
                loss += one_step_loss
        else:
            Q = model(states).max(1)[0]
            states.volatile = True
            target_Q = target_model(states).max(1)[0]
            target_Q[final_mask] = 0
            target_Q.volatile = False
            expected_Q = args.discount_factor * target_Q + rewards

            # Compute loss, backpropagate and apply clipped gradient update
            loss = value_criterion(Q, expected_Q)
            optimizer.zero_grad()
            loss.backward()
            for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        # Sync target network if needed
        if args.current_frame % args.sync_every == args.sync_every - 1:
            target_model.load_state_dict(model.state_dict())

        # Adapt if needed
        if args.noise == 'learned':
            model.adapt()

        # Update frame-level meters
        if args.noise == 'learned':
            args.losses.update(loss)
        else:
            args.losses.update(loss.data[0])
        args.rewards.update(float(reward))

        # Move on
        state = successor
        args.current_frame += 1

        # Update progress bar every frame
        args.bar.suffix = '({frame}/{size}) | Total: {total:} | ETA: {eta:} | AvgLoss: {loss:.4f} | AvgReward: {rewards: .4f}'.format(
                    frame=args.current_frame,
                    size=args.n_frames,
                    #data=data_time.avg,
                    #bt=batch_time.avg,
                    total=args.bar.elapsed_td,
                    eta=args.bar.eta_td,
                    loss=args.losses.avg,
                    rewards=args.rewards.avg)
        args.bar.next()

    # Update episode-level meters
    args.returns.update(args.rewards.sum)
    args.episode_lengths.update(int(args.current_frame - initial_frame))

    args.bar.suffix += ' | Total Loss {loss} | Return {return_} | Episode Length {length}\n'.format(
                loss=round(args.losses.sum, 4),
                return_=args.returns.val,
                length=args.episode_lengths.val)
    args.bar.next()


    # Initiate evaluation if needed
    if args.current_frame - args.eval_start > args.eval_every:
        args.test_time = True

    return env, model, target_model, optimizer, args


def trainPPO(env, model, optimizer, value_criterion, policy_criterion, args):
    model.train()
    state = env.reset()
    return env, model, optimizer, args
