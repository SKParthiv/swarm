# Swarm Information Thresholds

This repository studies how decentralized swarm performance degrades as shared information is throttled:

- **Information axis:** \( I(t) \) measured as compressed bits per second.
- **Performance axis:** \( P(t) \) normalized between blind and oracle baselines.

The initial implementation target is **Level 1 (MPE `simple_spread_v3`)** with a lightweight algorithmic controller so the information-throttling framework can be validated immediately.

## Phase 1 Decisions (Level 1: `simple_spread_v3`)

### Environment and Swarm

- [x] PettingZoo MPE `simple_spread_v3`
- [x] Homogeneous sparse swarm (3 identical agents, 3 landmarks)
- [x] Minimum safety layer via stalling/deadlock penalties
- [x] Monte Carlo seed ensembles

### Baseline Normalization \( P(t) \)

- [x] Empirical normalizer with blind/oracle anchors
- [x] **Blind baseline \(P=0.0\):** greedy local-only policy
- [x] **Oracle baseline \(P=1.0\):** centralized optimal landmark assignment
- [x] **Driving policy selected for Phase 1:** Dynamic Average Consensus (algorithmic, no MARL training dependency)

Why this choice: the immediate objective is validating the Information Throttling Engine and mapping phase-space trajectories. A consensus + heuristic setup removes RL training noise and makes degradation attributable to information constraints.

### Information Throttling \( I(t) \)

- [x] Dimensionality truncation
- [x] Resolution quantization
- [x] Temporal sparsity / frequency throttling
- [ ] Final choice of entropy estimator \(H(S)\) implementation

### Remaining TBD (Phase 1)

- [ ] Penalty coefficients \((\alpha, \beta, \gamma)\) for completion time, kinetic impact, and stalling.
- [ ] Final entropy-calculation method for \(H(S)\).

## Level 1 Benchmark Class: Spatial Coordination

Spatial coordination has low \(I(t)\) demand: agents distribute over stationary landmarks and avoid collisions. Coordination can remain viable under strongly compressed, low-frequency teammate information.

![2D benchmark environments](https://github.com/user-attachments/assets/6967190b-3411-40ed-be53-87214604bffb)

| PettingZoo Environment | Task Class | Oracle Baseline (P=1.0) | Blind Baseline (P=0.0) |
| --- | --- | --- | --- |
| MPE `simple_spread` | Spatial Coordination | Global-state awareness; flawless minimum-time landmark coverage. | Zero inter-agent communication; isolated local sensing only. |
| SISL `pursuit` | Dynamic Interception | Perfect, zero-latency coordination and maximum encirclement. | Uncoordinated pursuit with duplicate targeting and low capture rate. |
| MPE `simple_speaker_listener` | Asymmetric Navigation | Unrestricted bandwidth between speaker and listener for flawless pathing. | Zero bits transmitted; listener behavior collapses to random wandering. |
| SISL `waterworld` | Multi-role Foraging | Maximum safe food collection with high-rate teammate trajectory sharing. | Frequent poison/teammate collisions due to purely local field-of-view behavior. |

For Level 1:

- **Blind:** no inter-agent communication, greedy nearest-landmark behavior.
- **Oracle:** complete global state and optimal one-to-one assignment.
- **Throttled:** decentralized consensus using throttled shared intent signals.

## Boilerplate Project Structure

```text
swarm/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── level1_simple_spread.yaml
└── src/
    └── swarm/
        ├── __init__.py
        ├── env/
        │   ├── __init__.py
        │   └── simple_spread_v0.py
        ├── policies/
        │   ├── __init__.py
        │   ├── blind_greedy.py
        │   ├── oracle_assigner.py
        │   └── dynamic_average_consensus.py
        ├── throttling/
        │   ├── __init__.py
        │   ├── observation_filters.py
        │   └── bitrate.py
        └── experiments/
            ├── __init__.py
            └── level1_runner.py
```

## Quick Start

1. Install dependencies:

   ```bash
   pip install -e .
   ```

2. Run Level 1 with the consensus policy:

   ```bash
   python -m swarm.experiments.level1_runner --config configs/level1_simple_spread.yaml --policy consensus
   ```

3. Run baseline anchors:

   ```bash
   python -m swarm.experiments.level1_runner --config configs/level1_simple_spread.yaml --policy blind
   python -m swarm.experiments.level1_runner --config configs/level1_simple_spread.yaml --policy oracle
   ```

## Phase 2+ (Roadmap)

- **Level 2:** Dynamic interception (e.g., SISL `pursuit`, MPE `simple_tag`)
- **Level 3:** Asymmetric/heterogeneous communication tasks (e.g., `simple_speaker_listener`, `waterworld`)
- **Phase 3:** 3D/ROS2+Gazebo port while preserving the \(I(t)\) vs \(P(t)\) normalization framework
