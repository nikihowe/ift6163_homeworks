import torch

from collections import OrderedDict

import ift6163.infrastructure.pytorch_util as ptu

from .base_agent import BaseAgent
from ift6163.models.ff_model import FFModel
from ift6163.policies.MLP_policy import MLPPolicyPG, MLPPolicyAC
from ift6163.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.utils import *
from ift6163.agents.pg_agent import PGAgent


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']

        self.pg_agent = PGAgent(env, agent_params)  # Niki added

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        discount_every = terminal_n.shape[0]
        # print("discount every", discount_every)

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):
            model = self.dyn_models[i]

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful

            indices = np.arange(num_data)
            shuffled_indices = np.random.choice(indices, num_data_per_ens)

            observations = ob_no[shuffled_indices]
            actions = ac_na[shuffled_indices]
            next_observations = next_ob_no[shuffled_indices]

            # Copy this from previous homework
            log = model.update(observations, actions, next_observations, self.data_statistics)
            loss = log['Training Loss']
            losses.append(loss)

        # DONE Pick a model at random
        current_model_idx = np.random.choice(self.ensemble_size)
        current_model = self.dyn_models[current_model_idx]

        # DONE Use that model to generate one additional next_ob_no
        # for every state in ob_no (using the policy distribution)
        # Hint: You may need the env to label the rewards
        # Hint: Keep things on policy

        # Get the actions according to the current model
        chosen_actions = self.actor.get_action(ob_no)

        # print("replay buffer")
        # print("replay obs", self.replay_buffer.obs.shape)
        #
        # print("obs", ob_no.shape)
        # print("chosen actions", chosen_actions.shape)

        # Get the rewards from those states and actions
        rewards_from_chosen_actions, _ = self.env.get_reward(ob_no, chosen_actions)
        terminal_indices = np.array([i * 500 - 1 for i in range(1, int(discount_every / 500) + 1)])
        # print("terminal indices", terminal_indices)

        rewards_from_chosen_actions = rewards_from_chosen_actions
        terminals_from_chosen_actions = np.zeros_like(rewards_from_chosen_actions)
        terminals_from_chosen_actions[terminal_indices] = 1
        # print("replay terminals", self.replay_buffer.terminals.shape)
        # print("are there nonzero?", np.where(self.replay_buffer.terminals != 0))
        # print("rewards from chosen actions", rewards_from_chosen_actions.shape)
        # print("terminals from chosen actions", terminals_from_chosen_actions.shape)

        # Get the estimated next states from those states and actions
        estimated_next_ob_no = current_model.get_prediction(ob_no, chosen_actions, self.data_statistics)

        # print("estimated next ob no", estimated_next_ob_no.shape)

        # take a random choice of the different models we have
        # get model (model-based) from previous homework

        # DONE add this generated data to the real data
        assert ob_no.shape[0] == chosen_actions.shape[0] == rewards_from_chosen_actions.shape[0] == \
               estimated_next_ob_no.shape[0] == terminals_from_chosen_actions.shape[0]

        new_paths = []
        # for i in range(chosen_actions.shape[0]):
        # print("adding terminals", terminals_from_chosen_actions[i])
        # print("adding obs", ob_no[i])
        # raise SystemExit
        new_paths.append(
            {
                'observation': np.array(ob_no),
                'action': np.array(chosen_actions),
                'next_observation': np.array(estimated_next_ob_no),
                'terminal': np.array(terminals_from_chosen_actions),
                'reward': np.array(rewards_from_chosen_actions)
            }
        )
        # print("appended the new paths once")
        # print("the terminals are nonzero at", np.where(new_paths[0]['terminal'] == 1.)[0])
        # print("obs", new_paths[0]['observation'].shape)
        # print("action", new_paths[0]['action'].shape)
        # print("next obs", new_paths[0]['next_observation'].shape)
        # print("ter", new_paths[0]['terminal'].shape)
        # print("reward", new_paths[0]['reward'].shape)
        self.add_to_replay_buffer(new_paths)
        # print("added to the replay buffer")

        # DONE Perform a policy gradient update
        ob_recent, ac_recent, re_recent, ne_ob_recent, te_recent = \
            self.sample(ob_no.shape[0])
        ob_recent = ptu.from_numpy(ob_recent)
        ac_recent = ptu.from_numpy(ac_recent)
        ne_ob_recent = ptu.from_numpy(ne_ob_recent)

        # print("we sampled")

        # We use the PGAgent just to calculate the advantages
        estimated_q_vals = self.pg_agent.calculate_q_vals(re_recent[np.newaxis])
        estimated_advantages = self.pg_agent.estimate_advantage(ob_recent, re_recent, estimated_q_vals, te_recent)

        loss = OrderedDict()
        loss['Critic_Loss'] = self.critic.update(ob_recent, ac_recent, ne_ob_recent, re_recent, te_recent)

        loss['Actor_Loss'] = self.actor.update(ptu.to_numpy(ob_recent), ptu.to_numpy(ac_recent),
                                               estimated_advantages, estimated_q_vals)
        # loss['FD_Loss'] = np.mean(losses)
        return loss['Actor_Loss']

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_recent_data(  # used to be sample_random_data (changed to keep on policy)
            batch_size * self.ensemble_size)
