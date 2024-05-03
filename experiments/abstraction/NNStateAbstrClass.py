# Python imports.
import sys, os
import numpy as np
# Other imports.
import tensorflow as tf
from simple_rl.mdp import State
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
import numpy

# TODO:
    # Consider putting different MDP state abstractions into different directories in sa_models.
    # Add sampling to phi().

class NNStateAbstr(StateAbstraction):

    def __init__(self, abstraction_net, binary_classification=False):
        '''
        Args:
            abstraction_net (str): The name of the model.
        '''
        self.binary_classification = binary_classification
        self.abstraction_net = abstraction_net
    def phi(self, state: State):
        '''
        Args:
            state (simple_rl.State)

        Returns:
            state (simple_rl.State)
        '''
        pred = self.abstraction_net.predict(state.features().reshape(1, -1))
        pr_z_given_s = list(pred)
        # print("this is the pred", pred, "and the list", pr_z_given_s)
        
        if self.binary_classification:
            # print("this is the pred", pred)
            abstr_state_index = float(pred > 0.5)
        else:
            abstr_state_index = np.argmax(pr_z_given_s)
        # abstr_state_index = float(pred > 0.5)
        # print("best abstract index", abstr_state_index)
        return State(abstr_state_index)

    def phi_pmf(self, state):
        '''
        Args:
            state (simple_rl.State)

        Returns:
            (list): Contains probabilities. (index is z, value is prob of z given @state).
        '''
        return self.abstraction_net.predict([state])[0]
