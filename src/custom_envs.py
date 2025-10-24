# custom_envs.py
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv

class CustomCliffWalkingEnv(CliffWalkingEnv):
 
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, shape=(4, 12), cliff_coords=None, render_mode=None):

        super().__init__(render_mode=render_mode)

        self.shape = shape
        if cliff_coords is None:
            self.cliff_coords = [(shape[0] - 1, c) for c in range(1, shape[1] - 1)]
        else:
            self.cliff_coords = cliff_coords

        nrow, ncol = self.shape
        self.nS = nrow * ncol
        self.nA = 4

      
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def to_state(r, c): return r * ncol + c

        def next_state_reward_done(r, c, a):
            if a == 0: r = max(r - 1, 0)          
            elif a == 1: c = min(c + 1, ncol - 1) 
            elif a == 2: r = min(r + 1, nrow - 1) 
            elif a == 3: c = max(c - 1, 0)        
            s2 = to_state(r, c)
            done = (r, c) == (nrow - 1, ncol - 1)
            reward = 0 if done else (-100 if (r, c) in self.cliff_coords else -1)
            return s2, reward, done

        for r in range(nrow):
            for c in range(ncol):
                s = to_state(r, c)
                for a in range(self.nA):
                    s2, rew, done = next_state_reward_done(r, c, a)
                    self.P[s][a].append((1.0, s2, rew, done))


        self.isd = np.zeros(self.nS)
        self.isd[to_state(nrow - 1, 0)] = 1.0

 
        self.observation_space = gym.spaces.Discrete(self.nS)
        self.action_space = gym.spaces.Discrete(self.nA)

       
        self.s = to_state(nrow - 1, 0)
        self.render_mode = render_mode


    def render(self):
        if self.render_mode != "rgb_array":
            # optional text fallback
            grid = np.full(self.shape, " ", dtype="<U1")
            for r, c in self.cliff_coords: grid[r, c] = "C"
            grid[self.shape[0]-1, 0] = "S"
            grid[self.shape[0]-1, self.shape[1]-1] = "G"
            print("\n".join("".join(row) for row in grid))
            return

        cell = 30
        nrow, ncol = self.shape
        img = np.full((nrow*cell, ncol*cell, 3), 255, np.uint8)

        cliff = np.array([0, 0, 0], np.uint8)       # black
        start = np.array([0, 200, 0], np.uint8)     # green
        goal  = np.array([255, 215, 0], np.uint8)   # gold
        agent = np.array([0, 0, 255], np.uint8)     # blue

        for r, c in self.cliff_coords:
            img[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = cliff

        img[(nrow-1)*cell:nrow*cell, 0:cell] = start
        img[(nrow-1)*cell:nrow*cell, (ncol-1)*cell:ncol*cell] = goal

        if hasattr(self, "s"):
            r, c = divmod(self.s, ncol)
            img[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = agent

        return img

    # optional: update self.s on step (so the agent square renders)
    def step(self, action):
        probs = self.P[self.s][action][0] 
        _, s2, r, done = probs
        self.s = s2
        terminated = bool(done)
        truncated = False
        info = {}
        return self.s, r, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        nrow, ncol = self.shape
        self.s = (nrow - 1) * ncol + 0
        info = {}
        return self.s, info
