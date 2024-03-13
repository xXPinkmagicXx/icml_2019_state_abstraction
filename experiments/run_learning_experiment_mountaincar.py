# Python imports.
import sys
import random
import tensorflow as tf
import gym

# simple_rl imports.
from simple_rl.tasks import GridWorldMDP, GymMDP
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.agents import QLearningAgent, LinearQAgent, FixedPolicyAgent, RMaxAgent, RandomAgent
# from simple_rl.tasks import PuddleMDP
from simple_rl.run_experiments import run_agents_on_mdp, run_agents_lifelong, evaluate_agent, run_single_agent_on_mdp
from simple_rl.mdp import MDPDistribution

# Local imports.
from icml_2019_state_abstraction.experiments.abstraction.NNStateAbstrClass import NNStateAbstr
from icml_2019_state_abstraction.experiments.utils.experiment_utils import make_nn_sa
import icml_2019_state_abstraction.experiments.policies.CartPolePolicy as cpd
import icml_2019_state_abstraction.experiments.policies.mountaincar_pi_d as mpd


def diff_sampling_distr_experiment():
    '''
    Summary:
        Runs
    '''
    # Make MDP and Demo Policy.
    mdp_demo_policy_dict = {}

    env = GymMDP(env_name='CartPole-v0')
    
    # obs_size = env.get_num_state_feats()
    mdp_demo_policy_dict[env] = cpd.expert_cartpole_policy
    demo_agent = FixedPolicyAgent(cpd.expert_cartpole_policy)
    
    params = GridWorldMDP.get_parameters()

    # Make a NN for each sampling param.
    sampling_params = [0.0, 0.5, 1.0]

    test_mdp = GridWorldMDP() #
    agents = {"demo":demo_agent}
    sess = tf.Session()
    for epsilon in sampling_params:
        with tf.variable_scope('nn_sa' + str(epsilon), reuse=False) as scope:
            print("epsilon", epsilon)
            # tf.reset_default_graph()
            params["epsilon"] = epsilon
            abstraction_net = make_nn_sa(mdp_demo_policy_dict, sess, params, verbose=False)
            nn_sa = NNStateAbstr(abstraction_net)
            sa_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":env.get_actions(), "name":"$QL_\\phi-\\epsilon=" + str(epsilon) + "$"}, state_abstr=nn_sa)
            agents[epsilon] = sa_agent
    
    with tf.variable_scope('demo') as scope:
        abstraction_net_rand = make_nn_sa(mdp_demo_policy_dict, sess, params, verbose=False, sample_type="rand")
        nn_sa_rand = NNStateAbstr(abstraction_net_rand)
        sa_agent_rand = AbstractionWrapper(QLearningAgent, agent_params={"actions":env.get_actions(), "name":"$D \\sim U(S)$"}, state_abstr=nn_sa_rand, name_ext="")
        agents["rand"] = sa_agent_rand

    run_agents_on_mdp(agents.values(), test_mdp, instances=params['num_instances'], episodes=params['episodes'], steps=params['steps'], verbose=False)

    sess.close()

def get_params(env_name = "MountainCar-v0"):
    params={}
    params['multitask']=False
    params['obs_size']=4
    params['num_iterations_for_abstraction_learning']=500
    params['learning_rate_for_abstraction_learning']=0.001
    params['abstraction_network_hidden_layers']=2
    params['abstraction_network_hidden_nodes']=40
    params['num_samples_from_demonstrator']=5000
    params['episodes'] = 50
    params['env_name']= env_name
    params['steps']=10
    params['num_instances']=20
    params['rl_learning_rate']=0.001

    return params

def main():
    ## TODO: add mountain car sampling to make nn_state_abstraction work
    ## TODO: Make a random network and train that?
    ## TODO: Make a random network and train that?
    ## TODO: Given a random network, sample from that
    
    ## TODO: Train policy
    ## TODO: Train policy with abstraction
    ## TODO: Train policy with nn abstraction 
    ## TODO: Do we have a demonstrator policy? No
    ## TODO: initialize random weights? --> to predict and train

    # ======================
    # == Make Environment ==
    # ======================
    env_name = "MountainCar-v0"
    env = GymMDP(env_name)
    # env = gym.make(env_name)
    params = get_params()
    # num_test_mdps = 6 # 6 is max.
    mdp_demo_policy_dict = {}
    # obs_size = env.get_num_state_feats()
    mdp_demo_policy_dict[env] = mpd.expert_mountaincar_policy
    mdp = env 
    # ============================
    # == Make State Abstraction ==
    # ============================
    #nn_sa_file_name = "cartpole_nn_sa"
    sess = tf.Session()
    abstraction_net = make_nn_sa(mdp_demo_policy_dict, sess, params)
    nn_sa = NNStateAbstr(abstraction_net)

    # def get_discrete_state(state):
    #     discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    #     return tuple(discrete_state.astype(np.int))  # return as tuple
    # =================
    # == Make Agents ==
    # =================
    
    # Get actions and features
    actions = mdp.get_actions()
    actions =list(actions)
    #num_features = test_mdp.get_num_state_feats()
    # Make agents
    # linear_agent = LinearQAgent(actions=actions, num_features=num_features)
    params["explore"] = True
    ql_agent = QLearningAgent(actions)

    sa_agent = AbstractionWrapper(QLearningAgent, agent_params={"alpha":params['rl_learning_rate'],"epsilon":0.2,"actions":mdp.get_actions()}, state_abstr=nn_sa, name_ext="$-\\phi$")

    # ====================
    # == Run Experiment ==
    # ====================

    # if params['multitask']:
    #     run_agents_lifelong([sa_agent, linear_agent], test_mdp, samples=params['num_instances'], episodes=params['episodes'], steps=params['steps'], verbose=False)
    # else:
    #     # demo_agent = FixedPolicyAgent(cpd.expert_cartpole_policy)
    #     run_agents_on_mdp([sa_agent, linear_agent], test_mdp, instances=params['num_instances'], episodes=params['episodes'], steps=params['steps'], verbose=False)

    run_agents_on_mdp([sa_agent], mdp, instances=5, episodes=params['episodes'], steps=params['steps'], verbose=True)


if __name__ == "__main__":
    main()
    # diff_sampling_distr_experiment()
