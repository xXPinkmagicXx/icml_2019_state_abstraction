import os
from policies.PolicySB import PolicySB
ABSTRACTION = "icml"
def q_learning_function(policy: PolicySB):

    q_learning_agent = "Q-learning"
    env_name = "gym-" + policy.env_name 


    retrieve_folder = get_retrieve_folder_results(policy)
    file_name = policy.params['results_save_name'] + "_" + str(policy.seed) + ".csv" 


def get_retrieve_folder_results(policy: PolicySB):
    
    retrieve_folder = "results/" + "gym-" + policy.env_name + "/" + str(policy.policy_train_episodes) + "/"
    return retrieve_folder

def get_agent_name(policy: PolicySB):
    
    if policy.k_bins > 1:
        model = ABSTRACTION + "_" + str(policy.policy_train_episodes) + "_" + str(policy.k_bins) + "_" + policy.algo
    else:    
        model = ABSTRACTION + "_" + str(policy.policy_train_episodes) + "_" + policy.algo
    
    return model

def get_new_results_file_name(policy: PolicySB):

    new_save_folder = "results/" + ABSTRACTION + "/" + policy.env_name + "/" 
    model = get_agent_name(policy)
    new_save_name = model + "_" + str(policy.experiment_episodes) + "_" + str(policy.seed)
    
    if not os.path.exists(new_save_folder):
        os.makedirs(new_save_folder)
    
    return new_save_name

def get_new_save_folder(policy: PolicySB):
    
    new_save_folder = "results/" + ABSTRACTION + "/" + policy.env_name + "/" 
    return new_save_folder