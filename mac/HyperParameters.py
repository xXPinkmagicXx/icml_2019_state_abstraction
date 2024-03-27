## Make a class for the hyperparameters 



class MetaParameters:
    def __init__(self, env, env_name, max_learning_episodes, gamma, seed = 42) -> None:
        
        self.env = env
        self.env_name = env_name
        self.max_learning_episodes = max_learning_episodes
        self.seed = seed
        self.gamma = gamma
        self.state_dimension = len(env.reset())
    
    def to_Dictionary(self):
        
        meta_params = {}
        meta_params['env'] = self.env
        meta_params['env_name'] = self.env_name
        meta_params['seed'] = self.seed
        meta_params['gamma'] = self.gamma
        meta_params['max_learning_episodes'] = self.max_learning_episodes

        return meta_params

def make_parameters(action_space: int, state_dimension: int, actor_lr : list, critic_lr : list, critic_batch_size : list, critic_train_type : list, epsilon : list, max_buffer_size : list):
    
    actor_h = 40
    actor_num_h = 2
    critic_h = 40
    critic_num_h = 2
    critic_num_epochs = 10
    critic_target_net_freq = 1

    for ep in epsilon:
        for a_lr in actor_lr:
            for c_lr in critic_lr:
                for c_bs in critic_batch_size:
                    for c_tt in critic_train_type:
                        for mbs in max_buffer_size:
                                yield AlgorithmParameters(
                                                            max_buffer_size=mbs,
                                                            action_space=action_space,
                                                            state_dimension=state_dimension,
                                                            epsilon=ep,
                                                            actor_num_h=actor_num_h,
                                                            actor_h=actor_h,
                                                            actor_lr=a_lr,
                                                            critic_num_h=critic_num_h,
                                                            critic_h=critic_h,
                                                            critic_lr=c_lr,
                                                            critic_batch_size=c_bs,
                                                            critic_num_epochs=critic_num_epochs,
                                                            critic_target_net_freq=critic_target_net_freq,
                                                            critic_train_type=c_tt)     


class AlgorithmParameters:
    
    def __init__(
                self,
                max_buffer_size,
                action_space,
                state_dimension,
                epsilon,
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
        self.action_space = action_space
        self.state_dimension = state_dimension
        self.epsilon = epsilon
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
            alg_params['action_space'] = self.action_space
            alg_params['state_dimension'] = self.state_dimension
            alg_params['A'] = self.action_space
            alg_params['epsilon'] = self.epsilon
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