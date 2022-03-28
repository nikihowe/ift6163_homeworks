import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        ## DONE return the action that maxinmizes the Q-value
        # at the current observation
        action_q_values = self.critic.qa_values(observation)
        action = np.argmax(action_q_values)
        return action.squeeze()
