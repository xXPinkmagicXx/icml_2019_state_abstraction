# Python imports
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import time
import tensorflow_addons as tfa

# Other imports.
import tensorflow as tf
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, parent_dir)
sys.path.append("./policies")
sys.path.append("./abstraction")
# from simple_rl.tasks import PuddleMDP

# Local imports.
# Import policies.
import policies.Policy as Policy
import policies.CartPolePolicy as cpd

import Lunar_dqn.lunar_demonstrator as ld
from ..abstraction.abstraction_network import abstraction_network 
from ..abstraction.abstraction_network_new import abstraction_network_new 

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]



def plot_learned_abstraction(abstraction_net,num_clusters,iteration_number):
    colors=['red','green','blue','black','yellow','brown','orange']
    for _ in range(5):
        colors=colors+colors
    rg=np.arange(0,1,.01)
    samples=[]
    x_li=[]
    y_li=[]
    for x in rg:
        for y in rg:
            x_li.append(x)
            y_li.append(y)
            samples.append([x,y])
    p_li=abstraction_net.predict(samples)

    indices=[np.argmax([p]) for p in p_li]
    point_colors=[colors[i] for i in indices]


    plt.scatter(x_li,y_li,color=point_colors)
    if not os.path.exists("visuals"):
        # Make visuals dir if it's not there.
        os.makedirs("visuals")
    plt.savefig(os.path.join("visuals", "puddle"+str(iteration_number)+".pdf"))
    plt.close()

def enumeration_policy(size_z,size_a,num_mdps):
    a_in_z=np.zeros((size_z,num_mdps))
    for z in range(size_z):
        z_temp=z+0 #avoid copy by reference
        li=[]
        for _ in range(num_mdps):
            z_temp,r = divmod(z_temp, size_a)
            li.append(r)
        a_in_z[z,:]=list(reversed(li))
    return a_in_z

# ==========================================
# == Sampling Functions for Computing Phi ==
# ==========================================

# ------------
# -- Puddle --
# ------------

def collect_samples_from_demo_policy_puddle(mdp_demo_policy_dict, num_samples, epsilon=0.5):
    '''
    Args:
        mdp (simple_rl.MDP)
        num_samples (int)
        epsilon (float)

    Returns:
        (list): A collection of (s, a, mdp_id) tuples.

    Summary:
        Uses the demonstrator policy to collect the data.
    '''

    # Get MDP and demo policy.
    mdp = mdp_demo_policy_dict.keys()[0]
    demo_policy = mdp_demo_policy_dict[mdp]

    # Collect data set based on demo policy.
    cur_state = mdp.get_init_state()
    samples = []
    for _ in range(num_samples):
        demo_action = demo_policy(cur_state)
        if random.random() > epsilon:
            cur_a = demo_policy(cur_state)
        else:
            cur_a = random.randint(0, 1)

        action_index = mdp.get_actions().index(demo_action)
        samples.append((cur_state, action_index, 0))
        cur_state = mdp._transition_func(cur_state, cur_a)

    return samples

def collect_unif_random_samples_demo_policy_puddle(mdp_demo_policy_dict, num_samples):
    '''
    Args:
        mdp (simple_rl.MDP)
        demo_policy (lambda : simple_rl.State --> str)
        num_samples (int)

    Returns:
        (list): A collection of (s, a, mdp_id) tuples.

    Summary:
        Samples states uniformly at random from x in [0:1], y in [0:1].
    '''
    num_mdps=len(mdp_demo_policy_dict)
    samples = []
    mdp_init=mdp_demo_policy_dict.keys()[0]
    for _ in range(num_samples):
        cur_state = mdp_init.get_init_state()
        cur_state.x=np.random.random()
        cur_state.y=np.random.random()
        for (mdp_index,mdp) in enumerate(mdp_demo_policy_dict.keys()):
            demo_policy=mdp_demo_policy_dict[mdp]
            cur_a = demo_policy(cur_state)
            action_index = mdp.get_actions().index(cur_a)   
            samples.append(([cur_state.x, cur_state.y], action_index, mdp_index))

    return samples

# -----------
# -- Lunar --
# -----------

def collect_samples_from_demo_policy_random_s0_lunar(mdp_demo_policy_dict, num_samples):
    '''
    Args:
        mdp (simple_rl.MDP)
        demo_policy (lambda : simple_rl.State --> str)
        num_samples (int)

    Returns:
        (list): A collection of (s, a, mdp_id) tuples.
    '''
    samples=[]
    a_li = np.loadtxt("Lunar_dqn/a_li.txt")
    s_li = np.loadtxt("Lunar_dqn/s_li.txt")
    for index,(s,a) in enumerate(zip(s_li,a_li)):
        if index < num_samples:
            samples.append((s, a, 0))
    return samples

# --------------
# -- Cartpole --
# --------------

def collect_samples_from_demo_policy_random_s0_cartpole(mdp_demo_policy_dict, num_samples, epsilon=0.5):
    '''
    Args:
        mdp (simple_rl.MDP)
        num_samples (int)
        epsilon (float)

    Returns:
        (list): A collection of (s, a, mdp_id) tuples.
    '''

    num_mdps = len(mdp_demo_policy_dict)
    samples = []
    mdp = mdp_demo_policy_dict.keys()[0]
    demo_policy = mdp_demo_policy_dict[mdp]
    cur_state = mdp.env.reset() #mdp.get_init_state()
    for _ in range(num_samples):
        best_action = demo_policy(cur_state)
        if random.random() > epsilon:
            cur_a = demo_policy(cur_state)
        else:
            cur_a = random.randint(0, 1)

        action_index = mdp.get_actions().index(best_action)
        samples.append((cur_state, action_index, 0))
        cur_state, _, is_done, _ = mdp.env.step(cur_a)

        if is_done:
            cur_state = mdp.env.reset()
            
    return samples

def collect_unif_random_samples_demo_policy_cartpole(mdp_demo_policy_dict, num_samples):
    '''
    Args:
        mdp (simple_rl.MDP)
        num_samples (int)
        epsilon (float)

    Returns:
        (list): A collection of (s, a, mdp_id) tuples.
    '''

    num_mdps = len(mdp_demo_policy_dict)
    samples = []
    mdp = list(mdp_demo_policy_dict.keys())[0]
    demo_policy = mdp_demo_policy_dict[mdp]

    for _ in range(num_samples):
        cur_state = mdp.env.reset()
        cur_state = np.random.uniform(low=-4, high=4, size=(4,))
        mdp.env.state = cur_state

        # Get demo action.
        best_action = demo_policy(cur_state)
        #action_index = mdp.get_actions().index(best_action)
        samples.append((cur_state, best_action, 0))

    return samples

# -----------------
# -- Mountaincar --
# -----------------
def collect_unif_random_samples_demo_policy_mountaincar(mdp_demo_policy_dict, num_samples):
    '''
    Args:
        mdp (simple_rl.MDP)
        num_samples (int)
        epsilon (float)

    Returns:
        (list): A collection of (s, a, mdp_id) tuples.
    '''

    num_mdps = len(mdp_demo_policy_dict)
    samples = []
    # mdp = list(mdp_demo_policy_dict.keys())[0]
    # demo_policy = mdp_demo_policy_dict[mdp]

    # for _ in range(num_samples):
    #     cur_state = mdp.env.reset()
    #     cur_state = mdp.env.obseration.sample(1)
    #     mdp.env.state = cur_state

    #     # Get demo action.
    #     best_action = demo_policy(cur_state)
    #     action_index = mdp.get_actions().index(best_action)
    #     samples.append((cur_state, best_action, 0))

    return samples

# -------------
# -- Acrobot --
# -------------
def collect_unif_random_samples_demo_policy_acrobot(mdp_demo_policy_dict, num_samples):
    '''
    Args:
        mdp (simple_rl.MDP)
        num_samples (int)
        epsilon (float)

    Returns:
        (list): A collection of (s, a, mdp_id) tuples.
    '''

    num_mdps = len(mdp_demo_policy_dict)
    samples = []
    mdp = list(mdp_demo_policy_dict.keys())[0]
    demo_policy = mdp_demo_policy_dict[mdp]

    for _ in range(num_samples):
        cur_state = mdp.env.reset()
        cur_state = mdp.env.observation_space.sample()
        mdp.env.state = cur_state

        # Get demo action.
        best_action = demo_policy(cur_state)
        action_index = mdp.get_actions().index(best_action)
        samples.append((cur_state, best_action, 0))

    return samples 

# ===============================
# == Make NN State Abstraction ==
# ===============================

def make_nn_sa(mdp_demo_policy_dict, sess, params, verbose=True, sample_type="random"):
    '''
    Args:
        mdp_demo_policy_dict (dict):
            Key: (simple_rl.MDP)
            Val: (lambda : simple_rl.State --> str)
        sess (tf.session)
        params (dict)
        verbose (bool)
        sample_type (str): one of {"rand","demo"}

    Summary:
        Traing and saves a neural network state abstraction for the given
        @environment and @demo_policy.
    '''

    # MDP Specific parameters.
    num_mdps = len(mdp_demo_policy_dict)
    size_a = len(list(mdp_demo_policy_dict.keys())[0].get_actions())
    if verbose:
        print("this is the size of a", size_a)
    if num_mdps == 1:
        size_z = size_a
        a_in_z = np.array([x for x in range(size_z)]).reshape(size_z,num_mdps)
    else:
        size_z = np.power(size_a,num_mdps)
        a_in_z = enumeration_policy(size_z,size_a,num_mdps)

    num_abstract_states = size_z
    dir_path = os.path.dirname(os.path.realpath(__file__))
    abstraction_net = abstraction_network.abstraction_network(sess, params,num_abstract_states)
    sess.run(tf.compat.v1.global_variables_initializer())
    # saver = tf.train.Saver()
    if verbose:
        print("env_name:", params['env_name'])
    if params['env_name']=='PuddleMDP':
        if sample_type == "demo":
            # Sample from demo policy.
            samples_batch = collect_samples_from_demo_policy_puddle(mdp_demo_policy_dict, params['num_samples_from_demonstrator'])
        else:
            # Uniform random sampling.
            samples_batch = collect_unif_random_samples_demo_policy_puddle(mdp_demo_policy_dict, params['num_samples_from_demonstrator'])
    elif params['env_name']=='LunarLander-v2' or params['env_name']=='LunarNoShaping-v0':
        samples_batch = collect_samples_from_demo_policy_random_s0_lunar(mdp_demo_policy_dict, params['num_samples_from_demonstrator'])
    elif params['env_name']=='CartPole-v0':
        if sample_type == "demo":
            # Sample from demo policy.
            samples_batch = collect_samples_from_demo_policy_random_s0_cartpole(mdp_demo_policy_dict, params['num_samples_from_demonstrator'])
        else:
            # Uniform random sampling.
            samples_batch = collect_unif_random_samples_demo_policy_cartpole(mdp_demo_policy_dict, params['num_samples_from_demonstrator'])
    elif params['env_name'] == 'MountainCar-v0':
        
        samples_batch = collect_unif_random_samples_demo_policy_mountaincar(mdp_demo_policy_dict, params['num_samples_from_demonstrator'] )
    elif params['env_name'] == 'Acrobot-v1':
        samples_batch = collect_unif_random_samples_demo_policy_acrobot(mdp_demo_policy_dict, params['num_samples_from_demonstrator'])
    else:
        raise ValueError("(experiment_utils.py): make_nn_sa doesn't recognize given environment: `" + params["env_name"] + "'.")

    for iteration_number in range(params['num_iterations_for_abstraction_learning']):
        loss=abstraction_net.train(samples_batch, a_in_z)
        if verbose and iteration_number % 10 == 0:
            print("iteration number {} , obj {}".format(iteration_number,-loss))
        if iteration_number % 100 == 0 and params['env_name'] == 'PuddleMDP':
            plot_learned_abstraction(abstraction_net,size_z,iteration_number)
    return abstraction_net
      
def make_nn_sa_2(sess, params: dict, samples_batch, verbose=True):
    '''
    Args:
        :param sess (tf.session)
        :param params (dict)
        :param sample_batch (list of (state, action, mdp_id) tuples)
            Usually creted with saple_unif_random from the Policy or PolicySB class 
        :param verbose (bool)

    Summary:
        Traing and saves a neural network state abstraction for the given
        @environment and @demo_policy.
    '''

    print("Now making abstraction network...")
    assert type(samples_batch) == list, "samples_batch must be a list of (state, action, mdp_id) tuples."
    # MDP Specific parameters.

    num_mdps = params["num_mdps"]
    size_a = params["size_a"]
    
    if verbose:
        print("this is the size of a", size_a)
    
    if num_mdps == 1:
        size_z = size_a
        a_in_z = np.array([x for x in range(size_z)]).reshape(size_z,num_mdps)
    else:
        size_z = np.power(size_a,num_mdps)
        a_in_z = enumeration_policy(size_z,size_a,num_mdps)

    num_abstract_states = size_z
    print("This is the number of abstract states:", num_abstract_states)
    ## Create abstraction network
    abstraction_net = abstraction_network(sess, params, num_abstract_states)
    sess.run(tf.compat.v1.global_variables_initializer())
    
    ## print
    if verbose:
        print("env_name:", params['env_name'])

    ## Do training
    print("Now training the abstraction network...")
    for iteration_number in range(params['num_iterations_for_abstraction_learning']):
        
        print("With samples_batch like this:", random.choice(samples_batch))
        loss = abstraction_net.train(samples_batch, a_in_z)
        
        ## Print loss
        if verbose and iteration_number % 10 == 0:
            print("iteration number {} , obj {}".format(iteration_number,-loss))
    
    return abstraction_net
def make_nn_sa_3(params: dict, x_train, y_train, verbose=True):
    '''
    Args:
        :param sess (tf.session)
        :param params (dict)
        :param sample_batch (list of (state, action, mdp_id) tuples)
            Usually creted with saple_unif_random from the Policy or PolicySB class 
        :param verbose (bool)

    Summary:
        Traing and saves a neural network state abstraction for the given
        @environment and @demo_policy.
    '''

    num_mdps = params["num_mdps"]
    size_a = params["size_a"]
    
    if verbose:
        print("this is the size of a", size_a)
    
    if num_mdps == 1:
        size_z = size_a
        a_in_z = np.array([x for x in range(size_z)]).reshape(size_z,num_mdps)
    else:
        size_z = np.power(size_a,num_mdps)
        a_in_z = enumeration_policy(size_z,size_a,num_mdps)

    num_abstract_states = size_z
    print("This is the number of abstract states:", num_abstract_states)
    ## Create abstraction network
    abstraction_net = abstraction_network_new(params, num_abstract_states)
    
    ## print
    if verbose:
        print("env_name:", params['env_name'])

    ## Do training
    print("Now training the abstraction network...")
        
    # training the abstraction network
    tqdm_callback = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False, show_epoch_progress=False, leave_overall_progress=True, show_overall_progress=True)
    start_time = time.time()
    history = abstraction_net.net.fit(x_train, y_train, batch_size=32, callbacks=[tqdm_callback], epochs=params['num_iterations_for_abstraction_learning'])
    end_time = time.time()
    # plot history
    # plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # title_str = params['env_name'] + " Abstraction Accuracy" 
    # plt.title(title_str)
    # if os.path.exists(params['save_path']) == False:
    #     os.mkdir(params['save_path'])
    # plt.savefig(params['save_path'] + "/history.png")
    # print("Plot saved at:", params['save_path'] + "/history.png")

    # Save model
    abstraction_net_training_time = end_time - start_time
    abstraction_net.save_model(abstraction_net_training_time)
    

    return abstraction_net, abstraction_net_training_time