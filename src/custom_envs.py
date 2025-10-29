import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
import cv2

class CustomCliffWalkingEnv(CliffWalkingEnv):
    """
    Customizable CliffWalking environment with original-style textures,
    scalable grid size, and full RecordVideo compatibility.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, shape=(4, 12), cliff_coords=None, render_mode="rgb_array"):
        super().__init__(render_mode=render_mode)
        self.shape = shape
        self.render_mode = render_mode

        # Default cliff: bottom row except start and goal
        if cliff_coords is None:
            self.cliff_coords = [(shape[0] - 1, c) for c in range(1, shape[1] - 1)]
        else:
            self.cliff_coords = cliff_coords

        nrow, ncol = self.shape
        self.nS = nrow * ncol
        self.nA = 4

        # Transition probabilities
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def to_state(r, c): return r * ncol + c

        def next_state_reward_done(r, c, a):
            if a == 0:  # up
                r = max(r - 1, 0)
            elif a == 1:  # right
                c = min(c + 1, ncol - 1)
            elif a == 2:  # down
                r = min(r + 1, nrow - 1)
            elif a == 3:  # left
                c = max(c - 1, 0)
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

        self.s = to_state(nrow - 1, 0)  # Start position

        # Cell colors (RGB)
        self.color_floor = np.array([240, 240, 240], np.uint8)  # light gray
        self.color_cliff = np.array([30, 30, 30], np.uint8)      # dark gray/black
        self.color_start = np.array([0, 180, 0], np.uint8)       # green
        self.color_goal = np.array([255, 215, 0], np.uint8)      # gold
        self.color_agent = np.array([0, 0, 255], np.uint8)       # blue

        self.cell_size = 40  # pixels per grid square

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        nrow, ncol = self.shape
        self.s = (nrow - 1) * ncol + 0
        info = {}
        return self.s, info

    def step(self, action):
        transition = self.P[self.s][action][0]
        _, s2, r, done = transition
        self.s = s2
        terminated = bool(done)
        truncated = False
        info = {}
        return self.s, r, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            # text fallback
            grid = np.full(self.shape, " ")
            for r, c in self.cliff_coords:
                grid[r, c] = "C"
            grid[self.shape[0] - 1, 0] = "S"
            grid[self.shape[0] - 1, self.shape[1] - 1] = "G"
            print("\n".join("".join(row) for row in grid))
            return

        nrow, ncol = self.shape
        img = np.zeros(
            (nrow * self.cell_size, ncol * self.cell_size, 3), dtype=np.uint8
        )

        # Fill floor
        img[:] = self.color_floor

        # Draw cliffs
        for r, c in self.cliff_coords:
            y0, y1 = r * self.cell_size, (r + 1) * self.cell_size
            x0, x1 = c * self.cell_size, (c + 1) * self.cell_size
            img[y0:y1, x0:x1] = self.color_cliff

        # Draw start and goal
        s_r, s_c = nrow - 1, 0
        g_r, g_c = nrow - 1, ncol - 1
        img[s_r*self.cell_size:(s_r+1)*self.cell_size, s_c*self.cell_size:(s_c+1)*self.cell_size] = self.color_start
        img[g_r*self.cell_size:(g_r+1)*self.cell_size, g_c*self.cell_size:(g_c+1)*self.cell_size] = self.color_goal

        # Draw agent
        a_r, a_c = divmod(self.s, ncol)
        cy0, cy1 = a_r * self.cell_size, (a_r + 1) * self.cell_size
        cx0, cx1 = a_c * self.cell_size, (a_c + 1) * self.cell_size
        cv2.rectangle(img, (cx0 + 5, cy0 + 5), (cx1 - 5, cy1 - 5), self.color_agent.tolist(), -1)

        return img
