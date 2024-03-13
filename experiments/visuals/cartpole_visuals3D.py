# Python imports.
import tensorflow as tf
import gym
import itertools

# simple_rl imports.
from simple_rl.tasks import CartPoleMDP, GymMDP
from simple_rl.agents import RandomAgent
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.agents import QLearningAgent

# Local imports.
import run_learning_experiment_cartpole as rlec
from icml_2019_state_abstraction.experiments.abstraction.NNStateAbstrClass import NNStateAbstr
from icml_2019_state_abstraction.experiments.utils.experiment_utils import make_nn_sa,collect_samples_from_demo_policy_random_s0_cartpole
import icml_2019_state_abstraction.experiments.policies.CartPolePolicy as cpd
import icml_2019_state_abstraction.experiments.visuals.visual_utils as vu

def get_feature_dicts():
    '''
    Summary:
        Retrieves a list of feature dictionaries that contain each
        features index and name.
    '''
    loc_feature = {"name": "Location", "index": 0}
    vel_feature = {"name": "Velocity", "index": 1}
    ang_feature = {"name": "Angle", "index": 2}
    ang_vel_feature = {"name": "Angular Velocity", "index": 3}
    return [loc_feature, vel_feature, ang_feature, ang_vel_feature]


def main():

    # ======================
    # == Make Environment ==
    # ======================
    params = rlec.get_cartpole_params()
    num_test_mdps = 6 # 6 is max.
    mdp_demo_policy_dict = {}
    env = GymMDP(env_name='CartPole-v0')
    obs_size = env.get_num_state_feats()
    mdp_demo_policy_dict[env] = cpd.expert_cartpole_policy
    test_mdp = CartPoleMDP()

    # ============================
    # == Make State Abstraction ==
    # ============================
    sess = tf.Session()
    nn_sa_file_name = "cartpole_nn_sa"
    params['num_iterations_for_abstraction_learning']=500
    abstraction_net = make_nn_sa(mdp_demo_policy_dict, sess, params)
    nn_sa = NNStateAbstr(abstraction_net)

    # ====================================
    # == Visualize Abstract State Space ==
    # ====================================

    # Collect dataset based on learner.
    sa_agent = AbstractionWrapper(QLearningAgent, agent_params={"alpha":params['rl_learning_rate'],"epsilon":0.2,"actions":test_mdp.get_actions()}, state_abstr=nn_sa, name_ext="$-\\phi$")
    #visited_states = vu.collect_dataset(test_mdp, samples=2000) #, learning_agent=sa_agent)
    visited_states = collect_samples_from_demo_policy_random_s0_cartpole(mdp_demo_policy_dict, num_samples=2000)

    # Get feature indices.
    features = get_feature_dicts()

    # Visualize.
    vu.visualize_state_abstrs3D(visited_states, features, nn_sa)


if __name__ == "__main__":
    main()