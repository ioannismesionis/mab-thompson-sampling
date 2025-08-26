# Multi-Armed Bandits with Thompson Sampling

A comprehensive implementation and analysis of multi-armed bandit algorithms, focusing on Thompson Sampling and various exploration-exploitation strategies for recommendation systems and online decision making.

## 📚 Overview

This project explores the multi-armed bandit problem through the lens of recommendation systems, where an agent must balance **exploration** (trying new options) and **exploitation** (using known successful options) to maximize cumulative rewards over time.

## 🎯 Algorithms Implemented

### 1. Classic Bandit Algorithms (`multi_armed_bandits.ipynb`)
- **Random Agent**: Baseline random selection
- **Greedy Agent**: Always selects the best-performing arm
- **ε-Greedy Agent**: Balances exploration and exploitation with epsilon parameter
- **Softmax/Boltzmann Agent**: Probabilistic selection based on estimated rewards
- **Upper Confidence Bound (UCB)**: Optimistic selection using confidence intervals
- **LinUCB**: Contextual bandit with linear regression (demonstrated with MovieLens dataset)

### 2. Thompson Sampling (`thompson_sampling.ipynb`)
- **Bayesian Approach**: Uses Beta-Bernoulli conjugate priors
- **Prior Specification**: Demonstrates the importance of informed vs. uniform priors
- **Real-world Application**: Advertisement click-through rate optimization

## 🚀 Key Features

- **Comprehensive Comparison**: Side-by-side performance analysis of different algorithms
- **Real Data Integration**: MovieLens dataset for contextual recommendations
- **Bayesian Framework**: Proper treatment of uncertainty in Thompson Sampling
- **Visualization**: Extensive plots showing convergence, regret, and reward accumulation
- **Prior Analysis**: Detailed exploration of how domain knowledge improves performance

## 📊 Performance Metrics

Each implementation tracks:
- **Cumulative Reward**: Total rewards accumulated over time
- **Regret**: Difference from optimal performance
- **Arm Selection Frequency**: How often each option is chosen
- **Convergence Behavior**: How quickly algorithms learn optimal strategies

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mab-thompson-sampling.git
cd mab-thompson-sampling

# Install dependencies
pip install pandas numpy matplotlib scipy
# or using uv
uv sync
```

## 📖 Usage

Open and run the Jupyter notebooks:

1. **`multi_armed_bandits.ipynb`**: Start here for an introduction to MAB algorithms
2. **`thompson_sampling.ipynb`**: Deep dive into Bayesian Thompson Sampling

Each notebook is self-contained with explanations, implementations, and visualizations.

## 🎲 Multi-Armed Bandit Analogy

Think of a gambler in a casino with multiple slot machines (arms). Each machine has an unknown payout probability. The challenge is to figure out which machine is best while minimizing the money lost during the learning process.

In real applications, this translates to:
- **Web Advertising**: Which ad gets the most clicks?
- **Recommendation Systems**: Which items should we recommend?
- **A/B Testing**: Which website design performs better?
- **Clinical Trials**: Which treatment is most effective?

## 📈 Results Highlights

- Thompson Sampling with informed priors significantly outperforms uniform priors
- UCB provides excellent performance with theoretical guarantees
- Contextual approaches (LinUCB) leverage additional information effectively
- Proper exploration prevents getting stuck in suboptimal choices

## 🔗 References

- [Thompson, W. R. (1933). On the Likelihood that One Unknown Probability Exceeds Another](https://www.jstor.org/stable/2332286)
- [Li, L. et al. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146)
- [Russo, D. et al. (2018). A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038)
