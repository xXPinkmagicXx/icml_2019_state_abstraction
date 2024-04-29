# Python imports.
import sys, os
import random
import tensorflow as tf
import gymnasium as gym
from datetime import datetime
import time
# simple_rl imports.
from simple_rl.tasks import GymMDP
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper, ActionAbstraction
from simple_rl.agents import QLearningAgent, LinearQAgent, FixedPolicyAgent, RMaxAgent, RandomAgent
# from simple_rl.tasks import PuddleMDP
from simple_rl.run_experiments import run_agents_on_mdp, run_agents_lifelong, evaluate_agent, run_single_agent_on_mdp
from simple_rl.mdp import MDPDistribution
from simple_rl.mdp.StateClass import State
from ..mac.ActionWrapper import discretizing_wrapper
# Local imports.
# Import policies
import policies.Policy as Policy
import policies.PolicySB as PolicySB

import policies.CartPolePolicy as cpp
import policies.CartPolePolicySB as cpp_sb

import policies.MountainCarPolicy as mcp
import policies.MountainCarPolicySB as mcp_sb
import policies.MountainCarContinuousPolicy as mcpc
import policies.MountainCarContinuousPolicySB as mcpc_sb

import policies.AcrobotPolicy as abp
import policies.AcrobotPolicySB as abp_sb

import policies.LunarLanderPolicy as llp
import policies.LunarLanderPolicySB as llp_sb

import policies.PendulumPolicy as pp
import policies.PendulumPolicySB as pp_sb

import numpy as np
import pandas as pd
# abstraction
from .abstraction.NNStateAbstrClass import NNStateAbstr
from .utils.experiment_utils import make_nn_sa, make_nn_sa_2, make_nn_sa_3
from .abstraction.abstraction_network_new import abstraction_network_new

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_eager_execution()
# To make code compatible with old code implemented in tensorflow 1.x
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_policy_sb3(gym_env: GymMDP, algo: str, policy_train_steps: int, experiment_episodes: int, k_bins: int) -> PolicySB:
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
    6. MountainCarContinuous-v0
    """
    if gym_env.env_name == "MountainCar-v0":
        return mcp_sb.MountainCarPolicySB(gym_env, algo, policy_train_steps, experiment_episodes)
    

    if gym_env.env_name == "LunarLander-v2":
        return llp_sb.LunarLanderPolicySB(gym_env, algo, policy_train_steps, experiment_episodes)
    
    if gym_env.env_name == "CartPole-v0" or gym_env.env_name == "CartPole-v1":
        return cpp_sb.CartPolePolicySB(gym_env, algo, policy_train_steps, experiment_episodes)
    
    if gym_env.env_name == "Acrobot-v1":
        return abp_sb.AcrobotPolicySB(gym_env, algo, policy_train_steps, experiment_episodes)
    
    # --------- Countinuous action space environments --------- #
    if gym_env.env_name == "MountainCarContinuous-v0":
        return mcpc_sb.MountainCarContunuousPolicySB(gym_env, algo, policy_train_steps, experiment_episodes, k_bins)
    
    if gym_env.env_name == "Pendulum-v1":
        return pp_sb.PendulumPolicySB(gym_env, algo, policy_train_steps, experiment_episodes, k_bins)
    
    return NotImplementedError("Policy not implemented for this environment")

def get_mac_policy(gym_env: GymMDP, policy_time_episodes: int, experiment_episodes: int, k_bins: int) -> Policy:
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

    if gym_env.env_name == "CartPole-v0" or "CartPole-v1" == gym_env.env_name:
        return cpp.CartPolePolicy(gym_env, policy_time_episodes, experiment_episodes)

    if gym_env.env_name == "Acrobot-v1":
        return abp.AcrobotPolicy(gym_env, policy_time_episodes, experiment_episodes)

    if gym_env.env_name == "MountainCar-v0":
        return mcp.MountainCarPolicy(gym_env, policy_time_episodes, experiment_episodes)

    if gym_env.env_name == "LunarLander-v2":
        return llp.LunarLanderPolicy(gym_env, policy_time_episodes, experiment_episodes)
    # --------- Countinuous action space environments --------- #
    # TODO: Implement for MountainCarContinuous
    if gym_env.env_name == "MountainCarContinuous-v0":
        return mcpc.MountainCarContinuousPolicy(gym_env, policy_time_episodes, experiment_episodes, k_bins)
    if gym_env.env_name == "Pendulum-v1":
        return pp.PendulumPolicy(gym_env, policy_time_episodes, experiment_episodes, k_bins)

    return NotImplementedError("Policy not implemented for this environment")

def Get_GymMDP(env_name, k: int, render=False):
    """
    Args:
        :param env_name (str): Name of the environment
        :param k (int): Number of bins to discretize the action space into. Only used if the action space is continuous.
        :param render = False (bool): If True, sets the render_mode to human in the environment
    Returns:
        GymMDP object for the given environment
    Summary:
    This function creates a GymMDP object for the given environment.
    If the action space is continous, it discretizes the action space into k bin pr action.
    You can also set the render mode to human if you want to see the environment.
    """
    gym_env = gym.make(env_name, render_mode="human") if render else gym.make(env_name)
    ## Make the environment discrete
    if isinstance(gym_env.env.action_space, gym.spaces.Box):
        gym_env = discretizing_wrapper(gym_env, k)   
    
    gym_env = GymMDP(gym_env, render=False)
    
    return gym_env

def run_episodes_sb(env_name, policy: PolicySB, episodes=1, steps=500):
    """
    Args:
        :param gym_env (GymMDP)
        :param policy (Policy OR PolicySB)
    """
    print("Now running", episodes,"episodes of", env_name, "SB polic...")
    eval_env = Get_GymMDP(env_name=env_name, k = 20, render=True).env
    obs, info = eval_env.reset()
    for e in range(episodes):
        for _ in range(500):
            action = policy.expert_policy(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                obs, info = eval_env.reset()
                break
            eval_env.render()

def run_episodes_from_nn(env_name, abstraction_net: NNStateAbstr, episodes=1, steps=500):
    """
    Args:
        :param gym_env (GymMDP)
        :param policy (Policy OR PolicySB)
    """
    eval_env = Get_GymMDP(env_name, k = 20, render=True).env
    obs, info = eval_env.reset()
    for e in range(episodes):
        for s in range(steps):
            action = abstraction_net.phi(State(obs))
            obs, reward, terminated, truncated, info = eval_env.step(action.data)
            if terminated or truncated:
                print("terminated", terminated, "truncated", truncated, "after steps", s)
                obs, info = eval_env.reset()
                break
            eval_env.render()

def create_abstraction_network(policy, num_samples=10000, x_train=None):
    
    """
    Args:
        :param policy (PolicySB): Policy object
        :param x_train (np.array): Training data
        :param y_train (np.array): Labels
    Returns:
        NNStateAbstr object
    """
    start_time = time.time()
    x_train, y_train = policy.sample_training_data(num_samples)
    end_time = time.time()
    print("this is the time it took to sample the data", end_time - start_time)
    max_value = np.max(x_train)
    min_value = np.min(x_train)
    print("this is the max and min value", max_value, min_value)
    print("this si the shape of x_train", x_train[:2])
    print("this is the shape of y_train", y_train.shape, "with unique values", np.unique(y_train, return_counts=True))
    
    abstraction_net, abstraction_training_time = make_nn_sa_3(policy.params, x_train, y_train)
    
    StateAbsractionNetwork = NNStateAbstr(abstraction_net)
    
    return StateAbsractionNetwork, abstraction_training_time

def load_agent(env_name: str, algo: str, policy_train_episodes: int) -> NNStateAbstr:
    """
    Args:
        :param env_name (str): Name of the environment
        :param algo (str): Name of the algorithm
    Returns:
        NNStateAbstr object
    """
    print("loading pre-trained agent with algo", algo, "and environment", env_name)
    save_name = "trained-abstract-agents/"+ str(policy_train_episodes) + '/' + algo + "_" + env_name
    load_net =  tf.keras.models.load_model(save_name)
    nn_sa = NNStateAbstr(load_net)
    # Load training time
    if os.path.exists(save_name + "/abstraction_training_time.txt"):
        with open(save_name + "/abstraction_training_time.txt", "r") as f:
            abstraction_training_time = f.read()
    else:
        print("No training time found...")
        abstraction_training_time = 0

    print("loading complete...")
    return nn_sa, float(abstraction_training_time)

def get_policies(gym_env: str, algo: str, policy_train_episodes: int):
    """
    Args:
        :param gym_env (GymMDP)
        :param algo (str): Name of the algorithm
        :param policy_train_steps = 100_000 (int): Number of time steps the pre-trained policy was trained for
    Returns:
        Policy, PolicySB
    Summary:
    This function returns two policies, one for the given algorithm and the other for the MAC algorithm.
    if the algorithm is MAC, it returns only the MAC policy, and other as None.
    """
    policy_mac = get_policy(gym_env, policy_train_episodes)
    policy_mac.params["num_mdps"] = 1
    policy_mac.params["num_iterations_for_abstraction_learning"] = 100
    policy_mac.params["steps"] = 200
    policy_mac.params["episodes"] = 50
    
    if algo == "mac":
        return None, policy_mac
    
    policy = get_policy_sb3(gym_env, algo, policy_train_episodes)
    policy.params["num_mdps"] = 1
    policy.params["num_iterations_for_abstraction_learning"] = 100
    policy.params["steps"] = 200
    policy.params["episodes"] = 50

    return policy, policy_mac

def get_abstraction_networks(env_name: str, policySB: PolicySB, policy_mac: Policy, do_abstraction: bool, load_model: bool):
    """
    Summary:
        This function creates or loads the abstraction networks for the given environment.
        Can return None if no abstraction networks are created or loaded.
    Args:
        :param env_name (str): Name of the environment
        :param policy_mac (Policy): Policy object for MAC
        :param policySB (PolicySB): Policy object for the given algorithm
        :param do_abstraction (bool): If True, create abstraction networks
        :param load_model (bool): If True, load pre-trained abstraction networks
    Returns:
        abstraction_network (NNStateAbstr), abstraction_network_mac (NNStateAbstr)
    """


    abstraction_network = None
    abstraction_network_mac = None
    num_samples = 10000
    has_mac = policy_mac == None
    has_sb = policySB == None 
    if not has_mac and not has_sb:
        return ValueError("Both policies cannot be None")
    
    if load_model:
        print("Loading trained abstraction networks...")
        
        if has_mac:
            algo = policy_mac.params["algo"]
            abstraction_network_mac = load_agent(env_name, algo)
        if has_sb:
            algo = policySB.params["algo"]
            abstraction_network = load_agent(env_name, algo)    
        
        # returns can be None if no policy is provided 
        return abstraction_network, abstraction_network_mac

    if do_abstraction:
        print("Creating abstraction networks...")
        
        if has_mac:
            num_samples = policy_mac.params["num_samples_from_demonstrator"]
            abstraction_network_mac = create_abstraction_network_mac(policy_mac, num_samples)

        if policySB is not None and do_abstraction:
            num_samples = policySB.params["num_samples_from_demonstrator"]
            abstraction_network = create_abstraction_network(policySB, num_samples)
    
    else:
        print("No abstraction loaded or created, reuturns None, None...")
    
    return abstraction_network, abstraction_network_mac

def get_policy(gym_env: GymMDP, algo: str, policy_train_episodes: int, experiment_episodes: int, k_bins: int):
    if algo == "mac":
        policy = get_mac_policy(gym_env, policy_train_episodes, experiment_episodes, k_bins)
    else: 
        policy = get_policy_sb3(gym_env, algo, policy_train_episodes, experiment_episodes, k_bins)
    
    policy.params["num_mdps"] = 1
    policy.params["num_iterations_for_abstraction_learning"] = 11
    policy.params["steps"] = 20
    
    return policy



def main(env_name: str, algo: str, policy_train_episodes: int, experiment_episodes: int, k_bins=1, abstraction=True, load_model = False, run_expiriment=True,  verbose=False, seed=42):
    """
    Args:
        :param env_name (str): Name of the environment
        :param algo (str): Name of the algorithm
        :param abstraction = True (bool): If True, use state abstraction
        :param load_model = False (bool): If True, load a pre-trained abstraction agent
        :param verbose = False (bool) : If True, print the environment name
        :param seed = 42 (int) :Seed for reproducibility
    Returns:
        None
    Summary:
    This function runs the learning experiment for the given environment and does state
    abstraction if true.
    """

    verbose = True
    debug = True
    
    gym_env = Get_GymMDP(env_name, k = k_bins)
    ## Set seed
    # gym_env.env.seed(seed)
    random.seed(seed)

    ## Get actions and features
    actions = list(gym_env.get_actions())

    ## Get policies
    policy = get_policy(gym_env, algo, policy_train_episodes, experiment_episodes, k_bins)

    ## Get abstraction networks (can be none)
    if load_model:
        abstraction_network, abstraction_training_time = load_agent(env_name, algo, policy_train_episodes)
    elif abstraction:
        if debug:
            policy.params["num_samples_from_demonstrator"] = 100
        abstraction_network, abstraction_training_time = create_abstraction_network(policy, policy.params["num_samples_from_demonstrator"])
    else:
        print("No abstraction loaded or created...")

    ## Run one episode of the environment
    # run_episodes_sb(env_name, policy)
    # run_episodes_from_nn(env_name, abstraction_net=abstraction_network)
    # Make agents
    ## TODO: LinearQagent and number of features does not wor
    # num_features = gym_env.get_num_state_feats()
    # print("this is the number of features: ", num_features)
    demo_agent = FixedPolicyAgent(policy.demo_policy)
    ql_agent = QLearningAgent(actions=actions)
    # ql_agent = QLearningAgent(actions)
    agent_params = {"alpha":policy.params['rl_learning_rate'],"epsilon":0.1,"actions":actions}
    
    if abstraction_network is not None:
        sa_agent = AbstractionWrapper(QLearningAgent,
                                  agent_params=agent_params,
                                  state_abstr=abstraction_network,
                                  name_ext="_phi_"+ str(algo) + "_" + str(seed))
    else:
        print("skipping experiment for abstraction...")
    

    ## Agents in experiment
    agent_list = [sa_agent]
    if debug:
        policy.params['episodes'] = 5
    # Timestamp for saving the experiment
    # Run the experiment
    if run_expiriment:
        print("Running experiment...")
        # dir_for_plot = str(datetime.now().time()).replace(":", "_").replace(".", "_")
        dir_for_plot = str(policy_train_episodes)
        
        experiment_times = run_agents_on_mdp(agent_list,
                            gym_env,
                            instances=1,
                            episodes=policy.params['episodes'],
                            steps=policy.params['steps'],
                            verbose=True,
                            track_success=True,
                            success_reward=1,
                            dir_for_plot=dir_for_plot)
        
        get_and_save_results(policy, seed, training_time=experiment_times[0]+abstraction_training_time)
        
    else:
        print("Skipping experiment...")

def _read_file_and_get_results(file_path: str, episodes) -> list:
    """
    Args:
        :param file_path (str): Path to the file
    Returns:
        List 
    """
    with open(file_path, "r") as f:
        txt = f.read()
    
    return txt.split(",")[:episodes]
def get_and_save_results(policy: PolicySB, seed: int, training_time) -> None:
    q_learning_agent = "Q-learning"
    env_name = "gym-" + policy.env_name 
    episodes = policy.params['episodes']

    retrieve_folder = "results/" + env_name + "/" + str(policy.policy_train_episodes) + "/"
    file_name = q_learning_agent + "_phi_" + policy.algo + "_" + str(seed) + ".csv" 
    
    print("this is The retieve folder and file name", retrieve_folder, file_name)
    successes = _read_file_and_get_results(retrieve_folder + "success/" + file_name, policy.params['episodes'])
    times = _read_file_and_get_results(retrieve_folder + "times/" + file_name, policy.params['episodes'])
    rewards = _read_file_and_get_results(retrieve_folder + file_name, policy.params['episodes'])

    successes = [int(success) for success in successes]
    result = pd.DataFrame({"success": successes, "times": times, "rewards": rewards})
    # List of successes

    print("this is the success file txt:\n", "split into", successes)       

    ## Create save path
    ABSTRACTION = "icml"
    new_save_folder = "results/" + ABSTRACTION + "/" + policy.env_name + "/" 
    
    model = ABSTRACTION + "_" + str(policy.policy_train_episodes) + "_" + policy.algo
    new_save_name = model + "_" + str(episodes) + "_" + str(seed)
    if not os.path.exists(new_save_folder):
        os.makedirs(new_save_folder)
    
    result.to_csv(new_save_folder + new_save_name + ".csv")
    
    
    success_rate = np.mean(successes)

    result_info = pd.DataFrame({"success_rate": [success_rate], "training_time": [training_time], "episodes": [episodes], "seed": [seed], "agent": [model], })
    result_info.to_csv(new_save_folder + new_save_name + "_info.csv")

if __name__ == "__main__":
    ## Take in arguments from the command line.
    ## task to run
    environment = sys.argv[1]
    main(environment)
