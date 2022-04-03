import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import ift6163.util.class_util as classu

import numpy as np
import torch
from torch import distributions

from ift6163.infrastructure import pytorch_util as ptu
from ift6163.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    @classu.hidden_member_initialize
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 deterministic=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                            output_size=self._ac_dim,
                                            n_layers=self._n_layers,
                                            size=self._size)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                         self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           n_layers=self._n_layers, size=self._size)
            self._mean_net.to(ptu.device)
            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._logstd = nn.Parameter(
                    torch.zeros(self._ac_dim, dtype=torch.float32, device=ptu.device)
                )
                self._logstd.to(ptu.device)
                self._optimizer = optim.Adam(
                    itertools.chain([self._logstd], self._mean_net.parameters()),
                    self._learning_rate
                )

        if nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                n_layers=self._n_layers,
                size=self._size,
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    # DONE: got from HW3 (note it's different from HW1)
    # NOTE: also modified to have tanh output in deterministic case
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = torch.FloatTensor(observation)

        # DONE return the action that the policy prescribes
        if self._deterministic:
            action = ptu.to_numpy(self._mean_net(observation).detach())
            action = np.tanh(action)
        else:
            action = ptu.to_numpy(
                self(observation).sample().detach())  # NOTE: added sample because now it's a distribution here
        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        assert isinstance(observation, torch.FloatTensor)

        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if self._deterministic:
                ##  DONE output for a deterministic policy
                # action_distribution =
                mean = self._mean_net(observation)
                return mean
            else:
                batch_mean = self._mean_net(observation)
                scale_tril = torch.diag(torch.exp(self._logstd))
                batch_dim = batch_mean.shape[0]
                batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
                action_distribution = distributions.MultivariateNormal(
                    batch_mean,
                    scale_tril=batch_scale_tril,
                )
        return action_distribution


#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self._baseline_loss = nn.MSELoss()

    # DONE: got from hw3
    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # DONE: update the policy using policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
        # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
        # HINT4: use self.optimizer to optimize the loss. Remember to
        # 'zero_grad' first
        policy_dist_out = self(observations)
        log_prob_output = policy_dist_out.log_prob(actions)

        self.optimizer.zero_grad()  # DONE: ignore worrying about whether this is the correct place to put this
        loss = -torch.sum(log_prob_output * advantages)  # negative because we want to maximize
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            assert q_values is not None
            # DONE: update the neural network baseline using the q_values as
            # targets. The q_values should first be normalized to have a mean
            # of zero and a standard deviation of one. (DONE)

            # HINT1: use self.baseline_optimizer to optimize the loss used for
            # updating the baseline. Remember to 'zero_grad' first
            # HINT2: You will need to convert the targets into a tensor using
            # ptu.from_numpy before using it in the loss

            q_values_mean = np.mean(q_values)
            q_values_std = np.std(q_values)
            q_values_normalized = (q_values - q_values_mean) / (q_values_std + 1e-10)
            targets = ptu.from_numpy(q_values_normalized)

            baseline_estimates = self.baseline(observations).squeeze()  # squeeze to remove trailing singleton dimension

            self.baseline_optimizer.zero_grad()
            baseline_loss = self.baseline_loss(baseline_estimates, targets)
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }
        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self._baseline(observations)
        return ptu.to_numpy(pred.squeeze())


class MLPPolicyAC(MLPPolicy):
    # DONE: got from HW3
    def update(self, observations, actions, adv_n=None):
        # observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        assert adv_n is not None
        advantages = adv_n
        # advantages = ptu.from_numpy(adv_n)

        # DONE: update the policy using policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
        # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
        # HINT4: use self.optimizer to optimize the loss. Remember to
        # 'zero_grad' first
        policy_dist_out = self(observations)
        log_prob_output = policy_dist_out.log_prob(actions)

        self.optimizer.zero_grad()  # DONE: ignore if this is the correct place to put this
        loss = -torch.sum(log_prob_output * advantages)  # negative because we want to maximize
        loss.backward()
        self.optimizer.step()

        return loss.item()


class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """

    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)


class MLPPolicyDeterministic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """

    def __init__(self, *args, **kwargs):
        kwargs['deterministic'] = True
        super().__init__(*args, **kwargs)

    def update(self, observations, q_fun):
        # DONE: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss  # TODO: not sure how to avoid this
        self._optimizer.zero_grad()
        observations = ptu.from_numpy(observations)
        actions = self(observations)
        tanh_actions = torch.tanh(actions)
        loss = -torch.mean(q_fun.q_net(observations, tanh_actions))
        loss.backward()
        self._optimizer.step()

        return loss.item()
