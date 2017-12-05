# NoisyReproduce
A comparison of parameter noise methods for exploration in deep reinforcement learning


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
- Develop AdaptNoisyLinear layer
- Implement PPO and MuJoCo env handling
- Implement plotting (matplotlib is in Logger object, maybe try out visdom?)
- Test combinations of arguments (mostly envs with different )
- Experiments
  - DQN baseline
  - Noisy-DQN
  - Adapt-DQN
- Run baseline PPO, Noisy-PPO, and Adapt-PPO


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
