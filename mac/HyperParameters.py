## Make a class for the hyperparameters 



class MetaParameters:
    def __init__(self, env, max_learning_episodes) -> None:
        
        self.env = env
        self.env_name = env.env_name 
        self.max_learning_episodes = max_learning_episodes
    
    def to_Dictionary(self):
        
        meta_params = {}
        meta_params['env'] = self.env
        meta_params['env_name'] = self.env_name
        meta_params['max_learning_episodes'] = self.max_learning_episodes

        return meta_params

class AlgorithmParameters:
    
    def __init__(
                self,
                max_buffer_size,
                state_dimension,
                action_space,
                actor_num_h,
                actor_h,
                actor_lr,
                critic_num_h,
                critic_h,
                critic_lr,
                critic_batch_size,
                critic_num_epochs,
                critic_target_net_freq,
                critic_train_type):
        
        ## General
        self.max_buffer_size = max_buffer_size
        self.state_dimension = state_dimension
        self.action_space = action_space
        ## actor
        self.actor_num_h = actor_num_h
        self.actor_h = actor_h
        self.actor_lr = actor_lr
        ## critic
        self.critic_num_h = critic_num_h
        self.critic_h = critic_h
        self.critic_lr = critic_lr
        self.critic_batch_size = critic_batch_size
        self.critic_num_epochs = critic_num_epochs
        self.critic_target_net_freq = critic_target_net_freq
        self.critic_train_type = critic_train_type
    
    def to_Dictionary(self):
            
            alg_params = {}
            alg_params['max_buffer_size'] = self.max_buffer_size
            alg_params['state_dimension'] = self.state_dimension
            alg_params['action_space'] = self.action_space
            alg_params['actor_num_h'] = self.actor_num_h
            alg_params['actor_h'] = self.actor_h
            alg_params['actor_lr'] = self.actor_lr
            alg_params['critic_num_h'] = self.critic_num_h
            alg_params['critic_h'] = self.critic_h
            alg_params['critic_lr'] = self.critic_lr
            alg_params['critic_batch_size'] = self.critic_batch_size
            alg_params['critic_num_epochs'] = self.critic_num_epochs
            alg_params['critic_target_net_freq'] = self.critic_target_net_freq
            alg_params['critic_train_type'] = self.critic_train_type
            
            return alg_params

class HyperParameters:
	
    
    def __init__(self, meta_parameters : MetaParameters, algorithm_parameters :AlgorithmParameters) -> None:
        self.meta_parameters = meta_parameters
        self.algorithm_parameters = algorithm_parameters