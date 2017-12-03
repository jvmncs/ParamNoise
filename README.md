# NoisyReproduce
A submission to the ICLR Reproducibility Challenge

### Links to papers
Parameter Space Noise for Exploration : https://openreview.net/forum?id=ByBAl2eAZ&noteId=ByBAl2eAZ

Noisy Networks For Exploration : https://openreview.net/forum?id=rywHCPkAW&noteId=rywHCPkAW

### Resources
- [OpenAI Baselines](https://github.com/openai/baselines) for useful Atari wrappers
- [bearpaw's pytorch-classification repo](https://github.com/bearpaw/pytorch-classification) for utilities, logging, training framework
- [ikostrikov's PPO implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) for utilities and PPO guidance
- [pytorch-rl](https://github.com/jingweiz/pytorch-rl) for DQN guidance
- [Original DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) since both papers use the original hyperparameters, for the most part


### TODO

Immediate priority:
- Make DQN fully functional.
  - See TODOs in `main.py` for a few small, miscellaneous tasks
  - Bring in progress tracking from pytorch-classification repo.  Follow his lead [here](https://github.com/bearpaw/pytorch-classification/blob/cb5431ee091ec5e6f5225eb5bff7b918248b31d6/cifar.py#L268-L281), but with our metrics.
  - Finish `trainDQN` in `train.py`.  This should include the optimization step, as well as computing and tracking any necessary metrics (in particular, we want to be logging `['Episode', 'Frame', 'Episode Length', 'Loss', 'Reward']`).  Use `AverageMeter` and `Logger` classes, defined in `main.py` and passed in via `args`.
  - Finish `testDQN` in `test.py`.  This should be fairly straightforward once trainDQN is finished, since we'll want to be tracking the same metrics.  Probably best to create separate logger/meters for that.

Next Steps (these should happen in parallel):
- Experiments
  - DQN baseline
  - Noisy-DQN
- Develop AdaptNoisyLinear layer

Finally:
- Run Adapt-DQN experiments
- Implement PPO and MuJoCo env handling
- Run baseline PPO, Noisy-PPO, and Adapt-PPO
- Write up brief report with results and conclusions (we may want to just focus on DQN if we run out of time; can always publish a paper for PPO since that's technically new research)


### Games to Test
Alien -> OpenAI's version helps them a lot, but Deepmind shows no improvement
Enduro -> Improvements from both methods
Seaquest -> OpenAI shows improvement, DeepMind shows loss in improvement
Space Invaders -> OpenAI shows small improvements, Deepmind shows big improvement
WizardOfWor -> OpenAI shows loss, DeepMind shows big gains


### MuJoCo enviroments to test
