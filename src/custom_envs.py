import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv

class CustomCliffWalkingEnv(CliffWalkingEnv):

    def __init__(self, shape=(4, 12), cliff_coords=None, render_mode="rgb_array"):
        super().__init__(render_mode=render_mode)
        self.shape = shape
        nrow, ncol = self.shape

        # Default L-shaped cliff if none custom provided
        #(row_index, column_index)
        #row_index = 0 at the top, nrow-1 at the bottom
        #column_index = 0 at the left, ncol-1 at the right
        
        #Shift horizontal cliff up to row 2
        #horiz = [(nrow - 2, c) for c in range(1, 8)]


        #Move vertical cliff from (1,7) to (2,5)
        #vert = [(r, 5) for r in range(nrow - 3, nrow)]

        if cliff_coords is None:
            horiz = [(nrow - 1, c) for c in range(1, 8)]
            vert = [(r, 7) for r in range(nrow - 3, nrow)]
            self.cliff_coords = set(horiz + vert)
        else:
            self.cliff_coords = set(cliff_coords)
        
        # Create cliff boolean array
        self._cliff = np.zeros((nrow, ncol), dtype=bool)
        for r, c in self.cliff_coords:
            self._cliff[r, c] = True

        self.start_state = (nrow - 1, 0)
        self.goal_state = (0, ncol - 1)

        self.nS = nrow * ncol
        self.nA = 4

        def to_state(r, c):
            return r * ncol + c

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def next_state_reward_done(r, c, a):
            if a == 0: r = max(r - 1, 0)
            elif a == 1: c = min(c + 1, ncol - 1)
            elif a == 2: r = min(r + 1, nrow - 1)
            elif a == 3: c = max(c - 1, 0)

            s2 = to_state(r, c)
            goal = (r, c) == self.goal_state

            if goal: reward, done = 0, True
            elif (r, c) in self.cliff_coords:
                reward, done = -100, True
            else:
                reward, done = -1, False

            return s2, reward, done

        for r in range(nrow):
            for c in range(ncol):
                s = to_state(r, c)
                for a in range(self.nA):
                    ns, rew, done = next_state_reward_done(r, c, a)
                    self.P[s][a] = [(1.0, ns, rew, done)]

        self.observation_space = gym.spaces.Discrete(self.nS)
        self.action_space = gym.spaces.Discrete(self.nA)

        self.s = to_state(*self.start_state)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        nrow, ncol = self.shape
        self.s = (nrow - 1) * ncol + 0
        return self.s, {}

    def step(self, action):
        transition = self.P[self.s][action][0]
        _, s2, reward, done = transition

        if reward == -100:
            # Fall off cliff -> next timestep reset to start
            self.s = (self.shape[0] - 1) * self.shape[1] + 0
        else:
            self.s = s2

        return self.s, reward, done, False, {}
    
    


