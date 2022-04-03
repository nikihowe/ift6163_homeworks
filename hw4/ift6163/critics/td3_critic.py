from .ddpg_critic import DDPGCritic
import numpy as np
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy

from ift6163.infrastructure import pytorch_util as ptu
from ift6163.policies.MLP_policy import ConcatMLP


class TD3Critic(DDPGCritic):

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        # max_ac = ptu.from_numpy(max_action)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        q_t_values = self.q_net(ob_no, ac_na).squeeze()
        
        # TODO compute the Q-values from the target network 
        ## Hint: you will need to use the target policy
        next_action = self.actor_target.get_action(next_ob_no)
        eps = np.random.normal(0, self.td3_noise, size=next_action.shape)
        clipped_eps = np.clip(eps, -0.5, 0.5)  # clipped value taken from https://spinningup.openai.com/en/stable/_modules/spinup/algos/td3/td3.html
        # tanh_next_action = np.tanh(next_action)
        next_action = next_action + clipped_eps
        qa_tp1_values = self.q_net_target(next_ob_no, ptu.from_numpy(next_action)).squeeze()  # TODO: why?


        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma * qa_tp1_values * (1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        # self.learning_rate_scheduler.step()
        return {
            'Training Loss': ptu.to_numpy(loss),
        }


