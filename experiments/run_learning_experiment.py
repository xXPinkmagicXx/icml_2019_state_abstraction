# Python imports.
import sys
import random
import tensorflow as tf
import gym

# simple_rl imports.
from simple_rl.tasks import GymMDP
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.agents import QLearningAgent, LinearQAgent, FixedPolicyAgent, RMaxAgent, RandomAgent
# from simple_rl.tasks import PuddleMDP
from simple_rl.run_experiments import run_agents_on_mdp, run_agents_lifelong, evaluate_agent, run_single_agent_on_mdp
from simple_rl.mdp import MDPDistribution

# Local imports.
# Import policies
import policies.Policy as Policy
import policies.CartPolePolicy as cpp
import policies.MountainCarPolicy as mpd
import policies.AcrobotPolicy as abp

# abstraction
from abstraction.NNStateAbstrClass import NNStateAbstr
from utils.experiment_utils import make_nn_sa, make_nn_sa_2

import tensorflow as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_policy(gym_env: GymMDP):
    
    if gym_env.env_name == "CartPole-v0":
        return cpp.CartPolePolicy(gym_env)
    
    if gym_env.env_name == "Acrobot-v1":
        return abp.AcrobotPolicy(gym_env)

    if gym_env.env_name == "MountainCar-v0":
        return mpd.MountainCarPolicy(gym_env)

    return NotImplementedError("Policy not implemented for this environment")


def main(env_name, abstraction=True, verbose=False):

    ## get parameters
    gym_env = GymMDP(env_name)
    if verbose:
        print("this is the environment: ", gym_env.env_name)
    ## Get actions and features
    actions = list(gym_env.get_actions())
    
    ## Get policy
    policy = get_policy(gym_env)
    policy.params["num_mdps"] = 1
    policy.params["size_a"] = len(actions)
    
    ## Make Abstraction
    if abstraction:
        sess = tf.Session()
        sample_batch = policy.sample_unif_random()
        abstraction_net = make_nn_sa_2(sess, policy.params, sample_batch)
        nn_sa = NNStateAbstr(abstraction_net)

    # Make agents
    ## TODO: LinearQagent and number of features does not wor
    num_features = gym_env.get_num_state_feats()
    print("this is the number of features: ", num_features)
    linear_agent = QLearningAgent(actions=actions)
    # ql_agent = QLearningAgent(actions)
    agent_params = {"alpha":policy.params['rl_learning_rate'],"epsilon":0.2,"actions":actions}
    sa_agent = AbstractionWrapper(QLearningAgent,
                                  agent_params=agent_params,
                                  state_abstr=nn_sa,
                                  name_ext="-phi")

    ## Agents in experiment
    agent_list = [linear_agent, sa_agent]

    # Run the experiment
    run_agents_on_mdp(agent_list, gym_env, instances=20, episodes=policy.params['episodes'], steps=policy.params['steps'], verbose=True)


if __name__ == "__main__":
    ## Take in arguments from the command line.
    ## task to run
    environment = sys.argv[1]
    main(environment)
