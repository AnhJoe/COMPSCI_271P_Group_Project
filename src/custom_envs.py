import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv

# S = Start, G = Goal, C = Cliff
# . = Safe floor
# Change as needed for different layouts
LAYOUTS = {
    "CliffGauntlet": [
    "G............",
    ".............",
    ".CCCCCC......",
    ".CCCCCC......",
    ".............",
    "S............",
    ],
    "DoubleCanyon": [
    "....G.........",   
    "CCCC.CCC..CCCC",   
    "CC...CC...CCCC",   
    "CC.CCCC...CCCC",   
    "CC.C......CCCC",   
    "S.............",   
    ],
    "OpenDesert": [
    "......C..G...",
    ".C..C.....C.C.",
    "....CC........",
    "C......C..C...",
    "..C..C........",
    "...........C..",
    "S...C...C.....",
    ],
}

# Function to load a layout into a global variable
def load_layout(name):
    global ASCII_MAP
    ASCII_MAP = [row.replace(" ", "") for row in LAYOUTS[name]]

# Custom Cliff Walking Environment
class CustomCliffWalkingEnv(CliffWalkingEnv):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}
    
    # Load layout before creating an instance
    def __init__(self, render_mode="rgb_array"):
        # Ensure layout is loaded
        if "ASCII_MAP" not in globals():
            raise RuntimeError("Call load_layout(name) before creating CustomCliffWalkingEnv")
        
        # Initialize parent class
        super().__init__(render_mode=render_mode)
        self.render_mode = render_mode

        # Parse ASCII map
        self.rows = len(ASCII_MAP)
        self.cols = len(ASCII_MAP[0])
        self.cliff_coords = set()
        self.start = None
        self.goal = None
        for r in range(self.rows):
            for c in range(self.cols):
                ch = ASCII_MAP[r][c]
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
                elif ch == "C":
                    self.cliff_coords.add((r, c))
        if self.start is None or self.goal is None:
            raise ValueError("ASCII_MAP must contain exactly one S and one G.")

        # Updated observation space to dynamically match custom map (not the base 4x12)
        self.observation_space = gym.spaces.Discrete(self.rows * self.cols)
        self.action_space = gym.spaces.Discrete(4)

        # Set initial state index
        self.s = self.start[0] * self.cols + self.start[1]

        # Colors for rgb render
        self.cell = 50
        self.color_floor = (220, 220, 220)
        self.color_cliff = (60, 0, 0)
        self.color_start = (0, 200, 0)
        self.color_goal = (255, 215, 0)
        self.color_agent = (0, 0, 255)

    # Override reset and step to use custom layout
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = self.start[0] * self.cols + self.start[1]
        return self.s, {}

    # Override step function
    def step(self, action):
        
        # Convert integer state to grid position
        r, c = divmod(self.s, self.cols)

        # Movement logic
        if action == 0:   # up
            r -= 1
        elif action == 1: # right
            c += 1
        elif action == 2: # down
            r += 1
        elif action == 3: # left
            c -= 1

        # Clamp to edges
        r = max(0, min(self.rows - 1, r))
        c = max(0, min(self.cols - 1, c))

        # Goal check
        if (r, c) == self.goal:
            self.s = r * self.cols + c
            return self.s, 0, True, False, {}
        # Cliff check
        if (r, c) in self.cliff_coords:
            self.s = self.start[0] * self.cols + self.start[1]
            return self.s, -100, True, False, {}
        
        # Regular step
        self.s = r * self.cols + c
        return self.s, -1, False, False, {}

    # Override render to handle custom layout
    def render(self):
        if self.render_mode == "ansi":
            grid = [list(row) for row in ASCII_MAP]
            ar, ac = divmod(self.s, self.cols)
            if (ar, ac) not in self.cliff_coords and (ar, ac) != self.goal:
                grid[ar][ac] = "A"
            return "\n".join(" ".join(row) for row in grid)

        # rgb_array
        img = np.full((self.rows * self.cell, self.cols * self.cell, 3),
                      self.color_floor, dtype=np.uint8)

        # Cliffs
        for r, c in self.cliff_coords:
            img[r*self.cell:(r+1)*self.cell, c*self.cell:(c+1)*self.cell] = self.color_cliff

        # Start
        sr, sc = self.start
        img[sr*self.cell:(sr+1)*self.cell, sc*self.cell:(sc+1)*self.cell] = self.color_start

        # Goal
        gr, gc = self.goal
        img[gr*self.cell:(gr+1)*self.cell, gc*self.cell:(gc+1)*self.cell] = self.color_goal

        # Agent
        ar, ac = divmod(self.s, self.cols)
        img[ar*self.cell:(ar+1)*self.cell, ac*self.cell:(ac+1)*self.cell] = self.color_agent

        return img
