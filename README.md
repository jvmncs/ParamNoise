# ParamNoise
A comparison of parameter space noise methods for exploration in deep reinforcement learning

NOTE: This project is not maintained.  Reach out if you'd like to help reboot it.


### Links to papers
Parameter Space Noise for Exploration : https://openreview.net/forum?id=ByBAl2eAZ&noteId=ByBAl2eAZ

Noisy Networks For Exploration : https://openreview.net/forum?id=rywHCPkAW&noteId=rywHCPkAW


### Resources
- [OpenAI Baselines](https://github.com/openai/baselines) for useful Atari wrappers and replay buffer
- [bearpaw's pytorch-classification repo](https://github.com/bearpaw/pytorch-classification) for utilities, logging, training framework
- [ikostrikov's PPO implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) for other utilities and PPO guidance
- [pytorch-rl](https://github.com/jingweiz/pytorch-rl) for DQN help
- [PyTorch DQN tutorial](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) for PyTorch tricks
- [Original DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) since both papers use the original hyperparameters, for the most part


### TODOs
- Implement PPO and MuJoCo env handling
- Revisit logging; make sure everything is there to reproduce results in papers
- Implement plotting (matplotlib is in Logger object; maybe try out visdom)
- More tests (figure out different combinations of arguments to ensure everything's interacting well)
- Begin experiments (start with Mujoco; it's cheaper)


### Atari Games to Test
- Alien: Adaptive helps a lot, learned shows no improvement
- Enduro: Both methods improve
- Seaquest: Adaptive helps, learned performs worse than baseline
- Space Invaders: Adaptive helps, but learned helps more
- WizardOfWor: Adaptive worse than baseline, but learned helps a lot


### MuJoCo enviroments to test
- Hopper
- Walker2d
- HalfCheetah
- Sparse versions of these? (from rllab)
