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
from NNStateAbstrClass import NNStateAbstr
from experiment_utils import make_nn_sa, make_nn_sa_2
import policies.CartPolePolicy as cpp
import policies.mountaincar_pi_d as mpd


def diff_sampling_distr_experiment():
    '''
    Summary:
        Runs
    '''
    # Make MDP and Demo Policy.
    mdp_demo_policy_dict = {}

    env = GymMDP(env_name='CartPole-v0')

    # obs_size = env.get_num_state_feats()
    mdp_demo_policy_dict[env] = cpp.expert_cartpole_policy
    demo_agent = FixedPolicyAgent(cpp.expert_cartpole_policy)

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






def main(env_name, abstraction=True):

    ## get parameters
    gym_env = GymMDP(env_name)
    ## Get actions and features
    actions = list(gym_env.get_actions())
    
    ## Get policy
    policy = cpd.CartPolePolicy(gym_env)
    policy.params["num_mdps"] = 1
    policy.params["size_a"] = len(actions)
    
    ## Make Abstraction
    if abstraction:
        sess = tf.Session()
        sample_batch = policy.sample_unif_random()
        print("This is the sample batch", sample_batch)
        abstraction_net = make_nn_sa_2(sess, policy.params, sample_batch)
        nn_sa = NNStateAbstr(abstraction_net)

    # Make agents
    num_features = gym_env.get_num_state_feats()
    linear_agent = LinearQAgent(actions=actions, num_features=num_features)
    # ql_agent = QLearningAgent(actions)
    agent_params = {"alpha":policy.params['rl_learning_rate'],"epsilon":0.2,"actions":actions}
    sa_agent = AbstractionWrapper(QLearningAgent,
                                  agent_params=agent_params,
                                  state_abstr=nn_sa,
                                  name_ext="-phi")

    ## Agents in experiment
    agent_list = [linear_agent, sa_agent]

    # Run the experiment
    run_agents_on_mdp(agent_list, gym_env, instances=5, episodes=policy.params['episodes'], steps=policy.params['steps'], verbose=True)


if __name__ == "__main__":
    ## Take in arguments from the command line.
    ## task to run
    environment = sys.argv[1]
    main(environment)
