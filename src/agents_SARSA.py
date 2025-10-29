import numpy as np

class SarsaAgent:
    returns_next_action = True
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
        next_action = self.get_action(next_state)
        target = reward + self.gamma * self.Q[next_state, next_action] * (not done)
        
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        return next_action
