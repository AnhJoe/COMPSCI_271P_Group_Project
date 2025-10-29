import numpy as np

class QLearningAgent:
    def __init__(self, env, gamma=0.95, alpha=0.5, epsilon=1.0, decay_rate=0.995, min_eps=0.01):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_eps = min_eps
        n_states  = env.observation_space.n  
        n_actions = env.action_space.n 
        self.Q = np.zeros((n_states, n_actions))
        self.__init_logger()
        
    def __init_logger(self):
        self.rewards = []
        
    def epsilon_decay(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_eps)
                
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        else:
            return np.argmax(self.Q[state, :])

    
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = self.Q[next_state, :]
            next_q_max = max(next_qs)
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * next_q_max - self.Q[state, action])
