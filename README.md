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

# How to Run Analysis Only (no training):
The following commands will generate plots + markdown reports in:
Archive/<Mode>/<Layout>/

Baseline:
python main.py --mode Baseline --layout CliffGauntlet
python main.py --mode Baseline --layout DoubleCanyon
python main.py --mode Baseline --layout OpenDesert

Finetuned
python main.py --mode Finetuned --layout CliffGauntlet
python main.py --mode Finetuned --layout DoubleCanyon
python main.py --mode Finetuned --layout OpenDesert



# How to Train (only after uncommenting training block):
To train from scratch (and regenerate metrics):

1. Open src/main.py

2. Uncomment the TRAIN BOTH ALGORITHMS block

3. Then run:
python main.py --mode Baseline --layout CliffGauntlet --num-runs 50

## ⚙️ Run Locally
```bash
git clone 
cd cliffwalking_rl
pip install gymnasium numpy matplotlib
python train_sarsa.py
python train_qlearning.py


