import torch
import torch.nn as nn

from utils.norm import LayerNorm
from utils.noisy import NoisyLinear, AdaptNoisyLinear


class DQN(nn.Module):
    def __init__(self, action_space, noise = None):
        super(DQN, self).__init__()
        assert noise in [None, 'learned', 'adaptive']
        self.noise = noise
        self.action_dim = action_space.n

        self.conv1 = nn.Conv2d(4, 32, 8, 4) # output (b x 20 x 20 x 32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2) # output (b x 9 x 9 x 64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1) # outpus (b x 6 x 6 x 64)
        self.flattened_dim = 7*7*64
        if not self.noise:
            self.fc = nn.Linear(self.flattened_dim, 512)
            self.out = nn.Linear(512, self.action_dim)
        elif self.noise == 'adaptive':
            self.fc = AdaptNoisyLinear(self.flattened_dim, 512)
            self.ln = LayerNorm(512)
            self.out = AdaptNoisyLinear(512, self.action_dim)
        else:
            self.fc = NoisyLinear(self.flattened_dim, 512)
            self.out = NoisyLinear(512, self.action_dim)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.fc(x.view(x.size(0),-1))
        if self.noise == 'adaptive':
            x = self.ln(x)
        x = nn.functional.relu(x)
        return self.out(x)

    # Probably need to change these arguments to softmax(net_output) and softmax(perturbed_output)
    def adaptive_metric(self, net, perturbed):
        return nn.functional.kl_div(net, perturbed)

    def reset_noise(self):
        if self.noise:
            self.fc.reset_noise()
            self.out.reset_noise()


class PPO(nn.Module):
    def __init__(self, action_space, noise = None, clip_epsilon = 0.2, kind = 'large'):
        super(PPO, self).__init__()
        assert noise in [None, 'learned', 'adaptive']
        self.clip_epsilon = clip_epsilon
        self.action_dim = action_space.shape[0]

        # In the PPO original paper they say they use A3C network
        # both A3C network and DQN like network are used in the baselines and they default to large

        #if kind == 'small': # from A3C paper
        #   self.conv1 = nn.Conv2d(4, 16, 8, 4) # output (b x 20 x 20 x 16)
        #    self.conv2 = nn.Conv2d(16, 32, 4, 2) # output (b x 9 x 9 x 32)
        if kind == 'large': # Nature DQN
            self.conv1 = nn.Conv2d(4, 32, 8, 4) # output (b x 20 x 20 x 32)
            self.conv2 = nn.Conv2d(32, 64, 4, 2) # output (b x 9 x 9 x 64)
            self.conv3 = nn.Conv2d(64, 64, 3, 1) # outpus (b x 6 x 6 x 64)
        flattened_dim = 6 * 6 * 64

        if not self.noise:
            if action_space.__class__.__name__ == "Discrete": # if Atari
                self.fc = nn.Linear(flattened_dim, 512)
                self.out = nn.Linear(512, self.action_dim)
                self.critic = nn.Linear(512, 1)

            # TODO handle continuous case
            #else:
            #    self.fc = nn.Linear(flattened_dim, 512)
            #    self.out = nn.DiagGaussian(512, self.action_dim)
            #    self.critic = nn.Linear(512, 1)

        elif self.noise == 'adaptive':
            if action_space.__class__.__name__ == "Discrete": # if Atari
                self.fc = AdaptNoisyLinear(flattened_dim, 512)
                self.ln = LayerNorm(512)
                self.critic = AdaptNoisyLinear(512, 1)
                self.out = AdaptNoisyLinear(512, self.action_dim)

           # TODO handle continuous case
            #else:
            #    self.fc = AdaptNoisyLinear(flattened_dim, 512)
            #    self.ln = LayerNorm(512)
            #    self.out = DiagGaussian(512, self.action_dim)
            #    self.critic = AdaptNoisyLinear(512, 1)

        else:
            if action_space.__class__.__name__ == "Discrete": # if Atari
                self.fc = NoisyLinear(flattened_dim, 512)
                self.critic = NoisyLinear(512, 1)
                self.out = NoisyLinear(512, self.action_dim)

           # TODO handle continuous case
            #else:
                #self.fc = NoisyLinear(flattened_dim, 512)
                #self.critic = NoisyLinear(512, 1)
                #self.out = DiagGaussian(512, self.action_dim)


    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.fc(x)
        if self.noise == 'adaptive':
            x = self.ln(x)
        x = nn.functional.relu(x)

        return self.out(x), self.critic(x)

    def surrogate_loss(self, new, old, advantage):
        ratio = new / old
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
        return -torch.min(surr1, surr2).mean()

    def adaptive_metric(self, net, perturbed):
        ratio = perturbed / net
        return torch.abs(ratio - 1)

    def reset_noise(self):
        if noise:
            self.fc.reset_noise()
            self.out.reset_noise()
        else:
            pass


# taken from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/distributions.py

# TODO do the Guassian Output with noise
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        x = self.fc_mean(x)
        action_mean = x

        #  An ugly hack for my KFAC implementation.
        zeros = Variable(torch.zeros(x.size()), volatile=x.volatile)
        if x.is_cuda:
            zeros = zeros.cuda()

        x = self.logstd(zeros)
        action_logstd = x
        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        if deterministic is False:
            action = action_mean + action_std * noise
        else:
            action = action_mean
        return action

    def logprobs_and_entropy(self, x, actions):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()

        return action_log_probs, dist_entropy
