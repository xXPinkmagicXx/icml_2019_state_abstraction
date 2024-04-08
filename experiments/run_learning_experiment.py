# Python imports.
import sys
import random
import tensorflow as tf
import gym
from datetime import datetime
# simple_rl imports.
from simple_rl.tasks import GymMDP
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper, ActionAbstraction
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
import policies.LunarLanderPolicy as llps
import policies.PendulumPolicy as pp
# abstraction
from .abstraction.NNStateAbstrClass import NNStateAbstr
from .utils.experiment_utils import make_nn_sa, make_nn_sa_2

import tensorflow as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_policy(gym_env: GymMDP):
    """
    Args:
        gym_env (GymMDP)
    Returns:
        Policy

    Implemeted policies for the environments are
    1. CartPole-v0
    2. Acrobot-v1
    3. MountainCar-v0
    4. LunarLander-v2
    5. Pendulum-v1
    """
    if gym_env.env_name == "CartPole-v0":
        return cpp.CartPolePolicy(gym_env)
    
    if gym_env.env_name == "Acrobot-v1":
        return abp.AcrobotPolicy(gym_env)

    if gym_env.env_name == "MountainCar-v0":
        return mpd.MountainCarPolicy(gym_env)
    
    if gym_env.env_name == "LunarLander-v2":
        return llps.LunarLanderPolicy(gym_env)

    if gym_env.env_name == "Pendulum-v1":
        return pp.PendulumPolicy(gym_env)

    return NotImplementedError("Policy not implemented for this environment")


def main(env_name, abstraction=True, verbose=False, seed=42):
    """
    Args:
        env_name (str): Name of the environment
        abstraction = True (bool): If True, use state abstraction
        verbose = False (bool) : If True, print the environment name
        seed = 42 (int) :Seed for reproducibility
    Returns:
        None 
    This function runs the learning experiment for the given environment.
    """
    ## get parameters
    gym_env = GymMDP(env_name)
    
    ## Set seed
    gym_env.env.seed(seed)
    random.seed(seed)

    if verbose:
        print("this is the environment: ", gym_env.env_name)
    ## Get actions and features
    actions = list(gym_env.get_actions())
    
    
    ## Get policy
    policy = get_policy(gym_env)
    policy.params["num_mdps"] = 1
    policy.params["size_a"] = len(actions)
    policy.params["num_iterations_for_abstraction_learning"] = 10
    policy.params["steps"] = 20
    
    ## Make Abstraction
    if abstraction:
        sess = tf.Session()
        sample_batch = policy.sample_unif_random()
        abstraction_net = make_nn_sa_2(sess, policy.params, sample_batch)
        nn_sa = NNStateAbstr(abstraction_net)

    # If the action space is continuous
    # if isinstance(gym_env.env.action_space, gym.spaces.Box):
    #     k = 20
    #     discretized_actions = gym.spaces.Discrete(k)
    #     action_abstraction = ActionAbstraction(discretized_actions)

    # Make agents
    ## TODO: LinearQagent and number of features does not wor
    # num_features = gym_env.get_num_state_feats()
    # print("this is the number of features: ", num_features)
    linear_agent = QLearningAgent(actions=actions)
    # ql_agent = QLearningAgent(actions)
    agent_params = {"alpha":policy.params['rl_learning_rate'],"epsilon":0.2,"actions":actions}
    sa_agent = AbstractionWrapper(QLearningAgent,
                                  agent_params=agent_params,
                                  state_abstr=nn_sa,
                                  name_ext="_phi"+ "_" + str(seed))

    ## Agents in experiment
    agent_list = [sa_agent]

    # Timestamp for saving the experiment
    dir_for_plot = str(datetime.now().time()).replace(":", "_").replace(".", "_")
    
    # mode
    # Run the experiment
    run_agents_on_mdp(agent_list,
                      gym_env,
                      instances=1,
                      episodes=policy.params['episodes'],
                      steps=policy.params['steps'],
                      verbose=True,
                      track_success=True,
                      success_reward=1,
                      dir_for_plot=dir_for_plot)


if __name__ == "__main__":
    ## Take in arguments from the command line.
    ## task to run
    environment = sys.argv[1]
    main(environment)
