# Trading Environment Architecture

<cite>
**Referenced Files in This Document**
- [xauusd_env.py](file://env/xauusd_env.py)
- [xauusd_env_aggressive.py](file://env/xauusd_env_aggressive.py)
- [realistic_execution.py](file://env/realistic_execution.py)
- [backtest_engine.py](file://backtest/backtest_engine.py)
- [train_ppo.py](file://train/train_ppo.py)
- [train_ppo_aggressive.py](file://train/train_ppo_aggressive.py)
- [README.md](file://README.md)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
This document describes the Gymnasium-compatible trading environment system that standardizes reinforcement learning interactions for XAUUSD trading strategies. It covers:
- Discrete action spaces (flat/long and smart aggressive with short/flat/long)
- Observation space construction using windowed feature matrices plus current position
- Reward architecture combining PnL, transaction costs, turnover penalties, flat penalties, stability bonuses, and stop-loss penalties
- State management including position tracking, equity compounding, and episode termination conditions
- A realistic execution layer modeling spread widening, slippage, commissions, market impact, adverse selection, and event-driven cost spikes
- Configuration options for different trading styles (conservative vs aggressive)
- How the environment abstracts market complexity into a clean RL interface

## Project Structure
The trading environment lives under env/, with two Gymnasium environments and an execution model. Training scripts demonstrate how to configure and run each environment. A backtesting engine provides additional realistic cost assumptions and metrics.

```mermaid
graph TB
subgraph "Environment Layer"
E1["XAUUSDTradingEnv<br/>Discrete(Flat/Long)"]
E2["XAUUSDTradingEnvAggressive<br/>Discrete(Short/Flat/Long)"]
EX["RealisticExecutionModel<br/>Spread/Slippage/Impact"]
end
subgraph "Training & Evaluation"
T1["train_ppo.py"]
T2["train_ppo_aggressive.py"]
B["RigorousBacktester"]
end
T1 --> E1
T2 --> E2
B --> EX
```

**Diagram sources**
- [xauusd_env.py:7-57](file://env/xauusd_env.py#L7-L57)
- [xauusd_env_aggressive.py:6-64](file://env/xauusd_env_aggressive.py#L6-L64)
- [realistic_execution.py:24-86](file://env/realistic_execution.py#L24-L86)
- [train_ppo.py:25-44](file://train/train_ppo.py#L25-L44)
- [train_ppo_aggressive.py:25-48](file://train/train_ppo_aggressive.py#L25-L48)
- [backtest_engine.py:24-71](file://backtest/backtest_engine.py#L24-L71)

**Section sources**
- [README.md:418-470](file://README.md#L418-L470)

## Core Components
- XAUUSDTradingEnv: A discrete long-only environment with actions Flat/Long, windowed observations, and a reward that includes PnL, trade costs, turnover penalty, flat penalty, and hold bonus.
- XAUUSDTradingEnvAggressive: A “smart aggressive” environment supporting Short/Flat/Long, wider observation windows, leverage, and stop-loss truncation.
- RealisticExecutionModel: A module estimating execution costs (spread, slippage, commission, market impact, adverse selection) and adjusting fill prices accordingly.
- RigorousBacktester: A backtesting framework applying conservative cost assumptions and computing comprehensive performance metrics.

Key responsibilities:
- Standardize RL interaction via reset(), step(), and info dictionaries
- Provide consistent observation shapes across strategies
- Encapsulate realistic costs and risk controls
- Expose configuration knobs for trading style tuning

**Section sources**
- [xauusd_env.py:7-118](file://env/xauusd_env.py#L7-L118)
- [xauusd_env_aggressive.py:6-144](file://env/xauusd_env_aggressive.py#L6-L144)
- [realistic_execution.py:24-229](file://env/realistic_execution.py#L24-L229)
- [backtest_engine.py:24-254](file://backtest/backtest_engine.py#L24-L254)

## Architecture Overview
The system exposes a clean Gymnasium interface while hiding market complexity behind well-defined state transitions and rewards. Training scripts instantiate environments with strategy-specific parameters and train policies that output discrete actions. The execution model can be used during evaluation or integrated into custom environments to simulate realistic fills.

```mermaid
sequenceDiagram
participant Trainer as "Training Script"
participant Env as "Gymnasium Env"
participant Policy as "RL Policy"
participant Exec as "Execution Model"
Trainer->>Env : reset()
Env-->>Trainer : obs
loop Steps
Trainer->>Policy : act(obs)
Policy-->>Trainer : action
Trainer->>Env : step(action)
Env->>Exec : estimate_execution_cost(order, market_state)
Exec-->>Env : total_cost, breakdown
Env->>Env : compute PnL, costs, penalties
Env-->>Trainer : next_obs, reward, terminated, truncated, info
end
```

**Diagram sources**
- [train_ppo.py:25-44](file://train/train_ppo.py#L25-L44)
- [train_ppo_aggressive.py:25-48](file://train/train_ppo_aggressive.py#L25-L48)
- [xauusd_env.py:70-118](file://env/xauusd_env.py#L70-L118)
- [xauusd_env_aggressive.py:81-144](file://env/xauusd_env_aggressive.py#L81-L144)
- [realistic_execution.py:88-199](file://env/realistic_execution.py#L88-L199)

## Detailed Component Analysis

### XAUUSDTradingEnv (Conservative, Long-Only)
- Action space: Discrete {0=Flat, 1=Long}
- Observation space: Box of shape (window * features + 1), where the last element is the current position
- State variables: time index t, position pos ∈ {0,1}, steps, equity
- Reward components:
  - PnL from previous position over current return
  - Trade cost proportional to position change
  - Turnover penalty proportional to position change
  - Flat penalty when new position is 0
  - Hold bonus when no position change
- Equity updates multiplicatively by (1 + reward)
- Episode termination: natural end at T or truncation after max_episode_steps

```mermaid
flowchart TD
Start(["Step Entry"]) --> MapAction["Map action to new_pos"]
MapAction --> Delta["delta = |new_pos - pos|"]
Delta --> Costs["trade_cost = cost*delta<br/>turnover_penalty = turnover_coef*delta"]
Costs --> PnL["pnl = pos * r[t]"]
PnL --> Penalties["flat_pen if new_pos==0<br/>hold_bonus if delta==0"]
Penalties --> Reward["reward = pnl - trade_cost - turnover_penalty - flat_pen + hold_bonus"]
Reward --> Equity["equity *= (1 + reward)"]
Equity --> UpdateState["pos = new_pos<br/>t += 1<br/>steps += 1"]
UpdateState --> Term{"t >= T or steps >= max_episode_steps?"}
Term --> |Yes| End(["Return terminated/truncated"])
Term --> |No| NextObs["obs = concat(window_features, pos)"] --> End
```

**Diagram sources**
- [xauusd_env.py:75-118](file://env/xauusd_env.py#L75-L118)

**Section sources**
- [xauusd_env.py:21-68](file://env/xauusd_env.py#L21-L68)
- [xauusd_env.py:70-118](file://env/xauusd_env.py#L70-L118)

### XAUUSDTradingEnvAggressive (Smart Aggressive)
- Action space: Discrete {0=Short, 1=Flat, 2=Long} mapped to positions {-1, 0, 1}
- Observation space: Box of shape (window * features + 1), last element is current position
- Additional state: entry_price (virtual), leverage, stop_loss_pct
- Reward components:
  - PnL scaled by leverage
  - Trade cost and optional turnover penalty
  - Optional flat penalty and hold bonus
  - Stop-loss penalty and forced close with truncation when SL hit
- Episode termination: natural end at T, truncation on SL hit or max_episode_steps

```mermaid
flowchart TD
StartA(["Step Entry"]) --> MapA["action -> pos (-1,0,1)"]
MapA --> DeltaA["delta = |new_pos - pos|"]
DeltaA --> CostA["trade_cost = cost*delta<br/>turnover_pen = turnover_coef*delta"]
CostA --> PnLA["raw_pnl = pos * r[t]<br/>pnl = raw_pnl * leverage"]
PnLA --> SLCheck{"In position and SL hit?"}
SLCheck --> |Yes| SLPenalty["sl_penalty = -0.05<br/>new_pos = 0<br/>truncated = True"]
SLCheck --> |No| Aux["flat_pen, hold_bon"]
SLPenalty --> RewardA["reward = pnl - trade_cost - turnover_pen - flat_pen + hold_bon + sl_penalty"]
Aux --> RewardA
RewardA --> UpdateA["equity *= (1+reward)<br/>pos=new_pos<br/>t+=1<br/>steps+=1"]
UpdateA --> TermA{"t>=T or steps>=max or truncated?"}
TermA --> |Yes| EndA(["Return"])
TermA --> |No| ObsA["obs = concat(window_features, pos)"] --> EndA
```

**Diagram sources**
- [xauusd_env_aggressive.py:86-144](file://env/xauusd_env_aggressive.py#L86-L144)

**Section sources**
- [xauusd_env_aggressive.py:20-79](file://env/xauusd_env_aggressive.py#L20-L79)
- [xauusd_env_aggressive.py:81-144](file://env/xauusd_env_aggressive.py#L81-L144)

### Realistic Execution Model
- Models spread widening under volatility and events
- Adds slippage with volatility scaling and order-type effects
- Includes commission, market impact for large orders, and adverse selection
- Provides statistics tracking and per-trade cost breakdowns
- Adjusts fill price based on side and total cost

```mermaid
classDiagram
class RealisticExecutionModel {
+base_spread
+base_slippage
+commission
+spread_vol_multiplier
+slippage_vol_multiplier
+market_impact_coefficient
+adverse_selection_cost
+estimate_execution_cost(order, market_state)
+execute_trade(order, market_state, entry_price)
+get_statistics()
}
class SlippageSimulator {
+avg_slippage
+volatility_scaling
+get_slippage(market_state)
}
RealisticExecutionModel --> SlippageSimulator : "optional usage"
```

**Diagram sources**
- [realistic_execution.py:24-229](file://env/realistic_execution.py#L24-L229)
- [realistic_execution.py:232-275](file://env/realistic_execution.py#L232-L275)

**Section sources**
- [realistic_execution.py:35-86](file://env/realistic_execution.py#L35-L86)
- [realistic_execution.py:88-199](file://env/realistic_execution.py#L88-L199)
- [realistic_execution.py:211-229](file://env/realistic_execution.py#L211-L229)

### Backtesting Engine
- Applies conservative cost assumptions (spread multiplier, slippage, commission)
- Tracks trades, equity curve, and computes comprehensive metrics (returns, drawdown, Sharpe, Sortino, Calmar, win rate, profit factor)
- Supports walk-forward validation for robust evaluation

```mermaid
flowchart TD
Init["Initialize with agent, data, config"] --> Loop["Iterate historical data"]
Loop --> Observe["_get_observation(idx)"]
Observe --> Act["agent.act(obs)"]
Act --> Execute{"Position change?"}
Execute --> |Yes| Close["Close old position<br/>apply exit cost"]
Execute --> |No| Skip["No trade"]
Close --> Open{"Open new position?"}
Open --> |Yes| Entry["Pay entry cost"]
Open --> |No| Skip
Entry --> Record["Record trade"]
Skip --> Record
Record --> Metrics["Compute metrics at end"]
Metrics --> Results["Return results"]
```

**Diagram sources**
- [backtest_engine.py:73-146](file://backtest/backtest_engine.py#L73-L146)
- [backtest_engine.py:202-254](file://backtest/backtest_engine.py#L202-L254)

**Section sources**
- [backtest_engine.py:32-71](file://backtest/backtest_engine.py#L32-L71)
- [backtest_engine.py:73-146](file://backtest/backtest_engine.py#L73-L146)
- [backtest_engine.py:219-254](file://backtest/backtest_engine.py#L219-L254)

### Training Integration
- Conservative training uses XAUUSDTradingEnv with a moderate window and explicit cost parameterization
- Aggressive training uses XAUUSDTradingEnvAggressive with a longer window, macro-aware features, and configurable leverage/SL
- Both use parallel environments and save checkpoints periodically

```mermaid
sequenceDiagram
participant T as "train_ppo.py / train_ppo_aggressive.py"
participant F as "make_features"
participant E as "Gymnasium Env"
participant V as "SubprocVecEnv"
participant M as "PPO"
T->>F : build features and returns
T->>E : instantiate env with params
T->>V : wrap envs
T->>M : learn(...)
M->>V : rollout
V->>E : step(action)
E-->>V : obs, reward, term, trunc, info
V-->>M : samples
```

**Diagram sources**
- [train_ppo.py:25-44](file://train/train_ppo.py#L25-L44)
- [train_ppo_aggressive.py:25-48](file://train/train_ppo_aggressive.py#L25-L48)

**Section sources**
- [train_ppo.py:25-93](file://train/train_ppo.py#L25-L93)
- [train_ppo_aggressive.py:25-124](file://train/train_ppo_aggressive.py#L25-L124)

## Dependency Analysis
- Training scripts depend on feature generation and environment instantiation
- Environments are independent of models; they expose standardized interfaces
- Execution model is decoupled and can be used in evaluation or extended into environments
- Backtester depends on agent interface and data; it does not require the RL environment directly

```mermaid
graph LR
Features["features.make_features"] --> TrainPPO["train_ppo.py"]
Features --> TrainPPOAgg["train_ppo_aggressive.py"]
TrainPPO --> EnvStd["XAUUSDTradingEnv"]
TrainPPOAgg --> EnvAgg["XAUUSDTradingEnvAggressive"]
Eval["Evaluation Scripts"] --> EnvStd
Eval --> EnvAgg
Backtest["RigorousBacktester"] --> Exec["RealisticExecutionModel"]
```

**Diagram sources**
- [train_ppo.py:8-9](file://train/train_ppo.py#L8-L9)
- [train_ppo_aggressive.py:8-9](file://train/train_ppo_aggressive.py#L8-L9)
- [backtest_engine.py:24-71](file://backtest/backtest_engine.py#L24-L71)

**Section sources**
- [train_ppo.py:8-44](file://train/train_ppo.py#L8-L44)
- [train_ppo_aggressive.py:8-48](file://train/train_ppo_aggressive.py#L8-L48)
- [backtest_engine.py:24-71](file://backtest/backtest_engine.py#L24-L71)

## Performance Considerations
- Window size affects memory and temporal context; larger windows provide more history but increase observation dimensionality
- Leverage amplifies both returns and risks; ensure stop-loss and truncation logic align with risk tolerance
- Transaction cost modeling should match expected live conditions; underestimating costs leads to overfitting in simulation
- Parallel environment count balances throughput and resource constraints
- Use conservative backtest assumptions to avoid optimistic bias

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- Observation shape mismatch: Ensure features matrix has correct dimensions and window slicing indices are valid
- Unexpected truncation: Check stop-loss thresholds and max_episode_steps settings in aggressive mode
- Low reward variance: Tune turnover penalty, flat penalty, and hold bonus to encourage meaningful exploration
- Execution cost discrepancies: Validate market_state inputs (volatility, spread, liquidity, event flags) passed to execution model
- Training instability: Reduce learning rate, adjust batch size, or reduce n_steps per update

**Section sources**
- [xauusd_env.py:49-68](file://env/xauusd_env.py#L49-L68)
- [xauusd_env_aggressive.py:55-79](file://env/xauusd_env_aggressive.py#L55-L79)
- [realistic_execution.py:88-199](file://env/realistic_execution.py#L88-L199)

## Conclusion
The trading environment system provides a clean, Gymnasium-standard interface for RL-based XAUUSD trading strategies. It offers two distinct environments:
- A conservative, long-only setup with explicit cost and stability incentives
- An aggressive, multi-directional setup with leverage and stop-loss safeguards

Both share consistent observation and state management patterns, enabling direct comparison across strategies. The realistic execution model adds practical cost realism for evaluation and deployment readiness. Configuration knobs allow tailoring behavior to different trading styles while keeping the RL interface stable and predictable.

[No sources needed since this section summarizes without analyzing specific files]