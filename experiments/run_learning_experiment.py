# Python imports.
import sys, os
import random
from matplotlib.pyplot import show
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
from .abstraction.abstraction_network_pytorch import abstraction_network_pytorch 

import tensorflow as tf
import keras
import torch

tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_eager_execution()
# To make code compatible with old code implemented in tensorflow 1.x
tf.keras.utils.disable_interactive_logging()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_policy_sb3(gym_env: GymMDP, algo: str, policy_train_steps: int, experiment_episodes: int, k_bins: int, seed: int) -> PolicySB:
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
        return mcp_sb.MountainCarPolicySB(gym_env, algo, policy_train_steps, experiment_episodes, seed)
    

    if gym_env.env_name == "LunarLander-v2":
        return llp_sb.LunarLanderPolicySB(gym_env, algo, policy_train_steps, experiment_episodes, seed)
    
    if gym_env.env_name == "CartPole-v0" or gym_env.env_name == "CartPole-v1":
        return cpp_sb.CartPolePolicySB(gym_env, algo, policy_train_steps, experiment_episodes, seed)
    
    if gym_env.env_name == "Acrobot-v1":
        return abp_sb.AcrobotPolicySB(gym_env, algo, policy_train_steps, experiment_episodes, seed)
    
    # --------- Countinuous action space environments --------- #
    if gym_env.env_name == "MountainCarContinuous-v0":
        return mcpc_sb.MountainCarContunuousPolicySB(gym_env, algo, policy_train_steps, experiment_episodes, k_bins, seed)
    
    if gym_env.env_name == "Pendulum-v1":
        return pp_sb.PendulumPolicySB(gym_env, algo, policy_train_steps, experiment_episodes, k_bins, seed)
    
    return NotImplementedError("Policy not implemented for this environment")

def get_mac_policy(gym_env: GymMDP, policy_time_episodes: int, experiment_episodes: int, k_bins: int, seed: int) -> Policy:
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
        return cpp.CartPolePolicy(gym_env, policy_time_episodes, experiment_episodes, seed)

    if gym_env.env_name == "Acrobot-v1":
        return abp.AcrobotPolicy(gym_env, policy_time_episodes, experiment_episodes, seed)

    if gym_env.env_name == "MountainCar-v0":
        return mcp.MountainCarPolicy(gym_env, policy_time_episodes, experiment_episodes, seed)

    if gym_env.env_name == "LunarLander-v2":
        return llp.LunarLanderPolicy(gym_env, policy_time_episodes, experiment_episodes, seed)
    # --------- Countinuous action space environments --------- #
    
    if gym_env.env_name == "MountainCarContinuous-v0":
        return mcpc.MountainCarContinuousPolicy(gym_env, policy_time_episodes, experiment_episodes, k_bins, seed)
    if gym_env.env_name == "Pendulum-v1":
        return pp.PendulumPolicy(gym_env, policy_time_episodes, experiment_episodes, k_bins, seed)

    return NotImplementedError("Policy not implemented for this environment")

def Get_GymMDP(env_name, k: int, seed: int, time_limit_sec=None, render=False):
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
    
    gym_env = GymMDP(
        gym_env=gym_env,
        render=render,
        seed=seed,
        time_limit_sec=time_limit_sec)
    
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

def run_episodes_from_nn(env_name, abstraction_net: NNStateAbstr, seed: int,episodes=1, steps=500, verbose=True):
    """
    Args:
        :param gym_env (GymMDP)
        :param policy (Policy OR PolicySB)
    """
    eval_env = Get_GymMDP(env_name, k = 20, seed=seed,  render=True).env
    obs, info = eval_env.reset()
    for e in range(episodes):
        for s in range(steps):
            action = abstraction_net.phi(State(obs))
            obs, reward, terminated, truncated, info = eval_env.step(action.data)
            if terminated or truncated:
                
                if verbose:
                    print("terminated", terminated, "truncated", truncated, "after steps", s)
                
                obs, info = eval_env.reset()
                break

            eval_env.render()

def create_abstraction_network(policy, num_samples=10000, x_train=None, verbose=True):
    
    """
    Args:
        :param policy (PolicySB): Policy object
        :param x_train (np.array): Training data
        :param y_train (np.array): Labels
    Returns:
        NNStateAbstr object
    """
    start_time = time.time()
    X, y = policy.sample_training_data(num_samples, verbose)
    end_time = time.time()
    if verbose:
        print("this is the time it took to sample the data", end_time - start_time)
    # max_value = np.max(x_train)
    # min_value = np.min(x_train)
    # print("this is the max and min value", max_value, min_value)
    # print("this si the shape of x_train", x_train[:2])
    # print("this is the shape of y_train", y_train.shape, "with unique values", np.unique(y_train, return_counts=True))
    abstraction_network = abstraction_network_pytorch(policy.params)
    n_epochs = policy.params['num_iterations_for_abstraction_learning']
    batch_size = 1
    X = torch.Tensor(X)
    # is expected by the CrossEntropy
    y = torch.LongTensor(y)
    # print("type of X:", type(X), X[:10])
    # print("type of y:", type(y), y[:10])

    start_time = time.time()
    batch_size = 32 
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = abstraction_network(Xbatch)
            ybatch = y[i:i+batch_size]
            # print("Xbatch", Xbatch, "ybatch", ybatch)
            # print("y_pred:", y_pred, "ybatch", ybatch)s
            loss = abstraction_network.loss_fn(y_pred, ybatch)
            # print("Epoch:", epoch,"y_pred", y_pred,"ybatch", ybatch ,"Loss:", loss.item())
            abstraction_network.optimizer.zero_grad()
            loss.backward()
            abstraction_network.optimizer.step()
        print("Epoch: ", epoch)


    torch.save(abstraction_network, policy.params['save_path'] + ".pth")
    with torch.no_grad():
        pred = abstraction_network(Xbatch)
        print("this is a prediction", pred)

    end_time = time.time()
    StateAbsractionNetwork = NNStateAbstr(abstraction_network)
    abstraction_training_time = end_time - start_time
    
    # self.net.save(self.save_path)	
    with open(policy.params['save_path'] + "/abstraction_training_time.txt", "w") as f:
        f.write(str(abstraction_training_time))
    # abstraction_net, abstraction_training_time = make_nn_sa_3(policy.params, x_train, y_train)
    
    # StateAbsractionNetwork = NNStateAbstr(abstraction_net)
    
    return StateAbsractionNetwork, abstraction_training_time

def load_agent(env_name: str, algo: str, policy_train_episodes: int, seed: int, verbose: bool) -> NNStateAbstr:
    """
    Args:
        :param env_name (str): Name of the environment
        :param algo (str): Name of the algorithm
    Returns:
        NNStateAbstr object
    """

    save_name = "trained-abstract-agents/"+ str(policy_train_episodes) + '/' + algo + "_" + env_name + "_" + str(seed)
    load_net =  keras.models.load_model(save_name+".keras")
    nn_sa = NNStateAbstr(load_net)
    # Load training time
    if os.path.exists(save_name + "/abstraction_training_time.txt"):
        with open(save_name + "/abstraction_training_time.txt", "r") as f:
            abstraction_training_time = f.read()
    else:
        if verbose:
            print("No training time found...")
        abstraction_training_time = 0
    if verbose:
        print("loading complete...")
    return nn_sa, float(abstraction_training_time)

def load_agent_pytorch(env_name: str, algo: str, policy_train_episodes: int, seed: int, verbose: bool, policy) -> NNStateAbstr:
    """
    Args:
        :param env_name (str): Name of the environment
        :param algo (str): Name of the algorithm
    Returns:
        NNStateAbstr object
    """

    save_name = "trained-abstract-agents/"+ str(policy_train_episodes) + '/' + algo + "_" + env_name + "_" + str(seed)
    load_net =  torch.load(save_name+".pth")
    print("This is type of load net", type(load_net))
    
    nn_sa = NNStateAbstr(load_net)
    
    # Load training time
    if os.path.exists(save_name + "/abstraction_training_time.txt"):
        with open(save_name + "/abstraction_training_time.txt", "r") as f:
            abstraction_training_time = f.read()
    else:
        if verbose:
            print("No training time found...")
        abstraction_training_time = 0
    
    if verbose:
        print("loading complete...")
    
    return nn_sa, float(abstraction_training_time)

def get_policy(gym_env: GymMDP, algo: str, policy_train_episodes: int, experiment_episodes: int, k_bins: int, seed: int):
    if algo == "mac":
        policy = get_mac_policy(gym_env, policy_train_episodes, experiment_episodes, k_bins, seed)
    else: 
        policy = get_policy_sb3(gym_env, algo, policy_train_episodes, experiment_episodes, k_bins, seed)
    policy.params["num_mdps"] = 1
    # environment can max run for 1000 steps (LunarLander-v2, MountainCarContinuous-v0)
    policy.params["steps"] = 1000
    return policy


def main(
        env_name: str,
        algo: str,
        policy_train_episodes: int,
        experiment_episodes: int,
        k_bins: int,
        time_limit_sec=None,
        abstraction=True,
        load_model = False,
        run_expiriment=True,
        load_experiment=False,
        render=True, 
        verbose=False,
        debug=False,
        seed: int = 42):
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
    if debug:
        verbose = True
        policy_train_episodes = 3
        
    gym_env = Get_GymMDP(env_name, k = k_bins, seed=seed, time_limit_sec=time_limit_sec)
    ## Set seed
    # gym_env.env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ## Get actions and features
    actions = list(gym_env.get_actions())

    ## Get policies
    policy = get_policy(gym_env, algo, policy_train_episodes, experiment_episodes, k_bins, seed)

    if load_model:
        abstraction_network, abstraction_training_time = load_agent_pytorch(env_name, algo, policy_train_episodes, seed, verbose, policy)
    elif abstraction:
        if debug:
            policy.params["num_samples_from_demonstrator"] = 100
            policy.params["num_iterations_for_abstraction_learning"] = 3
            policy.params["steps"] = 10
        abstraction_network, abstraction_training_time = create_abstraction_network(policy, policy.params["num_samples_from_demonstrator"], verbose)
    else:
        print("No abstraction loaded or created...")

    # Make agents
    ## TODO: LinearQagent and number of features does not wor
    # demo_agent = FixedPolicyAgent(policy.demo_policy)
    # ql_agent = QLearningAgent(actions)
    name_ext = "_phi_" + str(policy.k_bins) + "_" + str(algo) + "_" + str(seed) if k_bins > 1 else "_phi_" + str(algo) + "_" + str(seed) 
    load_agent_path = "models/icml/" + env_name + "/" + str(experiment_episodes) + "/" "Q-learning" + name_ext
    agent_params = {"alpha":policy.params['rl_learning_rate'],"epsilon":0.1,"actions":actions,"load": load_experiment ,"load_path":load_agent_path}
    
    if abstraction_network is not None:
        # include k_bins if the action space is discretized
        sa_agent = AbstractionWrapper(QLearningAgent,
                                  agent_params=agent_params,
                                  state_abstr=abstraction_network,
                                  name_ext=name_ext)
    else:
        if verbose:
            print("skipping experiment for abstraction...")
    

    ## Agents in experiment
    agent_list = [sa_agent]
    if debug:
        policy.params['episodes'] = 5
    # Timestamp for saving the experiment
    # Run the experiment
    if run_expiriment:
        if verbose:
            print("Running experiment...")
        # dir_for_plot = str(datetime.now().time()).replace(":", "_").replace(".", "_")
        dir_for_plot = str(policy_train_episodes)
        
        experiment_times = run_agents_on_mdp(
                            agent_list,
                            gym_env,
                            instances=1,
                            episodes=policy.params['episodes'],
                            steps=policy.params['steps'],
                            verbose=True,
                            track_success=True,
                            reset_at_terminal = True,
                            open_plot=False,
                            success_reward=1,
                            dir_for_plot=dir_for_plot)
        
        get_and_save_results(policy=policy,
                             seed=seed,
                             experiment_train_time=experiment_times[0],
                             abstraction_train_time=abstraction_training_time,
                             verbose=verbose)
    
        

    else:
        if verbose:
            print("Skipping experiment...")
    
    if (run_expiriment or load_model or abstraction) and render:
       run_episodes_from_nn(env_name, abstraction_network, seed=seed, steps=1000, verbose=verbose) 

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


def get_and_save_results(policy: PolicySB, seed: int, abstraction_train_time: float, experiment_train_time: float, verbose=True) -> None:
    q_learning_agent = "Q-learning"
    env_name = "gym-" + policy.env_name 
    episodes = policy.params['episodes']

    retrieve_folder = "results/" + env_name + "/" + str(policy.policy_train_episodes) + "/"
    file_name = policy.params['results_save_name'] + "_" + str(seed) + ".csv" 
    
    if verbose:
        print("this is The retieve folder and file name", retrieve_folder, file_name)
    
    successes = _read_file_and_get_results(retrieve_folder + "success/" + file_name, policy.params['episodes'])
    times = _read_file_and_get_results(retrieve_folder + "times/" + file_name, policy.params['episodes'])
    rewards = _read_file_and_get_results(retrieve_folder + file_name, policy.params['episodes'])
    steps = _read_file_and_get_results(retrieve_folder + "steps/" + file_name, policy.params['episodes'])
    
    successes = [int(success) for success in successes]
    steps = [int(step) for step in steps]
    # steps = [int(step) for step in steps]
    result = pd.DataFrame({"success": successes, "times": times, "rewards": rewards, "steps": steps})

    ## Create save path
    ABSTRACTION = "icml"
    new_save_folder = "results/" + ABSTRACTION + "/" + policy.env_name + "/" 
    
    model = ABSTRACTION + "_" + str(policy.policy_train_episodes) + "_" + policy.algo
    new_save_name = model + "_" + str(episodes) + "_" + str(seed)
    if not os.path.exists(new_save_folder):
        os.makedirs(new_save_folder)
    
    # save results
    result.to_csv(new_save_folder + new_save_name + ".csv")

    success_rate = np.mean(successes)

    policy_train_time = policy.get_policy_train_time()
    total_train_time = policy_train_time + abstraction_train_time + experiment_train_time
    print("Training time  Policy : ", round(policy_train_time,4) , " Abstraction : ", round(abstraction_train_time,4) , " Experiment: ", round(experiment_train_time, 4), " Totaltime: ", round(total_train_time,5))

    if verbose:
        print("Total training time", total_train_time)

    result_info = pd.DataFrame({
        "agent": [model], 
        "episodes": [episodes],
        "success_rate": [success_rate],
        "policy_train_time": [policy_train_time],
        "abstraction_train_time": [abstraction_train_time],
        "experiment_train_time": [experiment_train_time],
        "total_train_time": [total_train_time],
        "seed": [seed],
        "total_steps": [np.sum(steps)],
        })
    
    result_info.to_csv(new_save_folder + new_save_name + "_info.csv")

def Get_Success_Rate(policy, rewards, ) -> list:
    """
    Args:
        :param policy (PolicySB): Policy object
        :param seed (int): Seed for reproducibility
    """
    pass

if __name__ == "__main__":
    ## Take in arguments from the command line.
    ## task to run
    environment = sys.argv[1]
    main(environment)
