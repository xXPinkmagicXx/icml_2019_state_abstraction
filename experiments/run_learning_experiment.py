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
import policies.LunarLanderPolicy as llp
import policies.LunarLanderPolicySB as llp_sb
import policies.PendulumPolicy as pp
import policies.MountainCarPolicySB as mpd_sb
import policies.CartPolePolicySB as cpp_sb
from huggingface_sb3 import load_from_hub


# abstraction
from .abstraction.NNStateAbstrClass import NNStateAbstr
from .utils.experiment_utils import make_nn_sa, make_nn_sa_2

import tensorflow as tf
# To make code compatible with old code implemented in tensorflow 1.x
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_policy_sb3(gym_env: GymMDP):
    """
    Args:
        :param gym_env (GymMDP)
    Returns:
        Policy

    Implemeted policies for the environments are
    1. CartPole-v0
    2. Acrobot-v1
    3. MountainCar-v0
    4. LunarLander-v2
    5. Pendulum-v1
    """
    algo = "dqn"
    path_to_learned_policy = './rl-trained-agents/' + algo + "_" + gym_env.env_name
    
    if gym_env.env_name == "MountainCar-v0":
        return mpd_sb.MountainCarPolicySB(gym_env)
    if gym_env.env_name == "LunarLander-v2":
        return llp_sb.LunarLanderPolicySB(gym_env)
    
    return cpp_sb.CartPolePolicySB(gym_env, algo)


def get_policy(gym_env: GymMDP):
    """
    Args:
        :param gym_env (GymMDP) : Gym MDP object
        :param sb3 (bool): If True, use stable baselines 3
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
        return llp.LunarLanderPolicy(gym_env)

    if gym_env.env_name == "Pendulum-v1":
        return pp.PendulumPolicy(gym_env)

    return NotImplementedError("Policy not implemented for this environment")


def main(env_name: str, algo: str, abstraction=True, verbose=False, seed=42):
    """
    Args:
        :param env_name (str): Name of the environment
        :param algo (str): Name of the algorithm
        :param abstraction = True (bool): If True, use state abstraction
        :param verbose = False (bool) : If True, print the environment name
        :param seed = 42 (int) :Seed for reproducibility
    Returns:
        None
    This function runs the learning experiment for the given environment and does state 
    abstraction if true.
    """
    ## get parameters
    gym_env = GymMDP(env_name, render=True, render_every_n_episodes=2)

    ## Set seed
    # gym_env.env.seed(seed)
    random.seed(seed)

    if verbose:
        print("this is the environment: ", gym_env.env_name)
    ## Get actions and features
    actions = list(gym_env.get_actions())


    ## Get policy
    if algo == 'mac':
        policy = get_policy(gym_env)
    else:
        policy = get_policy_sb3(gym_env)

    # policy_mac = get_policy(gym_env)

    policy.params["num_mdps"] = 1
    policy.params["size_a"] = len(actions)
    policy.params["num_iterations_for_abstraction_learning"] = 200
    policy.params["steps"] = 200
    
    ## Make Abstraction
    if abstraction:
        sess = tf.compat.v1.Session()
        
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
    demo_agent = FixedPolicyAgent(policy.demo_policy)
    linear_agent = QLearningAgent(actions=actions)
    # ql_agent = QLearningAgent(actions)
    agent_params = {"alpha":policy.params['rl_learning_rate'],"epsilon":0.1,"actions":actions}
    
    sa_agent = AbstractionWrapper(QLearningAgent,
                                  agent_params=agent_params,
                                  state_abstr=nn_sa,
                                  name_ext="_phi"+ "_" + str(seed))
    # sa_agent_mac = AbstractionWrapper(QLearningAgent,
    #                               agent_params=agent_params,
    #                               state_abstr=nn_sa_mac,
    #                               name_ext="_phi"+ "_" + str(seed))

    ## Agents in experiment
    agent_list = [sa_agent]

    # Timestamp for saving the experiment
    dir_for_plot = str(datetime.now().time()).replace(":", "_").replace(".", "_")

    # test if the new sb3 works
    # if GET_STABLE_BASELINES:
    #     vec_env = policy.model.get_env()
    #     obs = vec_env.reset()
    #     for _ in range(1000):
    #         # print("this is the observation", obs)
    #         action, _states = policy.model.predict(obs, deterministic=True)
    #         obs, rewards, done, info = vec_env.step(action)
    #         vec_env.render("human")

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
