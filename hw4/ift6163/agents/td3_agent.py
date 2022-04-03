import numpy as np

from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.policies.MLP_policy import MLPPolicyDeterministic
from ift6163.critics.td3_critic import TD3Critic
import copy

from ift6163.agents.ddpg_agent import DDPGAgent

class TD3Agent(DDPGAgent):
    def __init__(self, env, agent_params):

        super().__init__(env, agent_params)
        
        self.q_fun = TD3Critic(self.actor, 
                               agent_params, 
                               self.optimizer_spec)

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator
            should have been advanced one step, and the replay buffer should contain
            one more transition. Note that self.last_obs must always
            points to the new latest observation.
        """
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = 0.05
        perform_random_action = np.random.random() < eps

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            previous_frames = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(previous_frames)

        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs

        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()