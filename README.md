# Cliff Walking RL

Reinforcement Learning project using the `CliffWalking-v0` environment from **Gymnasium**.  
Compares **SARSA** (on-policy) and **Q-Learning** (off-policy) for navigating a 4×12 gridworld with penalties for falling off the cliff.

---

## 🧠 Environment
- **Start:** Bottom-left  
- **Goal:** Bottom-right  
- **Reward:** –1 per step, –100 for the cliff, +0 at goal  
- **Goal:** Learn an optimal policy that balances exploration and safety.

---

## ⚙️ Run Locally
```bash
git clone 
cd cliffwalking_rl
pip install gymnasium numpy matplotlib
python train_sarsa.py
python train_qlearning.py
