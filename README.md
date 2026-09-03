# Swarm Information Thresholds: Project Roadmap

This repository maps the fundamental relationship between shared information entropy ($I(t)$) and decentralized multi-agent coordination performance ($P(t)$). The framework isolates the decision-making layer by assuming ideal computation, instant algorithm execution, and ideal sensor/actuator dynamics.

## Phase 1: Core Framework & 2D Simulation (Level 1)

**1. Environment & Swarm Setup**

* **Approved:**
* * [x] Use PettingZoo MPE `simple_spread_v3` for the initial spatial coordination task.




* **Reason:** The state representation is simple and purely kinematic, allowing us to test the absolute minimum sufficient teammate information required for pure spatial assignment.




* * [x] Configure a Homogeneous Sparse Swarm (3 identical agents, 3 stationary landmarks).




* **Reason:** All agents share identical physical dynamics with no specialized roles, representing the lowest baseline for information transmission since agents rarely encounter dense bottleneck regions.




* * [x] Implement a "Minimum Safety Layer" with Stalling rather than hard crashes.




* **Reason:** Hard crashes and low-level safety reflex loops introduce confounding physical noise; allowing agents to stall or deadlock smoothly captures spatial conflict (wasted time/energy) without abruptly terminating the episode.




* * [x] Use Monte Carlo Seed Ensembles ($K$ fixed environment seeds).




* **Reason:** Random initial positions introduce high variance (e.g., a lucky spawn makes a blind swarm look effective); running fixed seeds allows us to calculate robust mean performance or worst-case envelopes.






* **TBD:**
* * [ ] Specific coefficients for the penalty formula (Time to completion $\alpha$, Kinetic Impact Energy $\beta$, and Stalling Time $\gamma$).




* **The Dilemma:** We must balance the weights of these penalties so that $P(t)$ degrades smoothly based on lost time and wasted energy. If the stalling penalty ($\gamma$) is too high, it mimics an abrupt failure rather than a gradual performance curve.







**2. Baseline Normalization Module ($P(t)$)**

* **Approved:**
* * [x] Build an Empirical Normalizer (Oracle vs. Blind baselines).




* **Reason:** It is practically impossible to find a universal mathematical algorithm to normalize performance across any arbitrary task; running physical simulation extremes allows us to natively bake the swarm's physical limits into a dimensionless $0.0$ to $1.0$ scale.




* * [x] Establish the **Oracle Baseline ($P = 1.0$)**.




* **Reason:** Giving agents unrestricted communication and complete global state awareness provides the true mathematical ceiling for the specific task and swarm combination.




* * [x] Establish the **Blind Baseline ($P = 0.0$)**.




* **Reason:** Forcing agents to use zero inter-agent communication and rely purely on isolated local sensing provides the absolute floor—how well the task can be done by sheer luck or isolated individual effort.






* **TBD:**
* * [ ] Selection of the specific agent driving policy.




* **The Dilemma:** We must choose between formal Dec-POMDP solvers, Multi-Agent Reinforcement Learning (MAPPO/QMIX), or Dynamic Average Consensus protocols. MARL allows for complex learned behaviors but requires training time, whereas classical consensus algorithms are faster to implement but may not scale to highly complex asymmetrical tasks.







**3. Information Throttling Engine ($I(t)$)**

* **Approved:**
* * [x] Define the X-axis universally as compressed bits per second (bps) using $X = H(S) \times f$.




* **Reason:** This converts arbitrary metrics (like roles, paths, and positions) into a single, uniform, algorithm-agnostic currency, allowing us to map an objective phase space trajectory regardless of hardware or task.




* * [x] Implement Observation Wrappers to modulate Dimensionality, Resolution, and Frequency.




* **Reason:** Treating the sensing stack as a black box and intercepting the observation space allows us to act directly as a throttle, forcing the swarm into low-info or high-info regimes by masking vectors, quantizing floats to 8-bit grids, or dropping update rates to 1Hz.






* **TBD:**
* * [ ] Selection of the exact method to calculate theoretical bits/Shannon Entropy $H(S)$.




* **The Dilemma:** Using a standard software compression library (like Python's gzip) introduces algorithmic biases. We need to decide whether to write a pure mathematical entropy calculation (measuring minimum binary questions) or rely on a standard lossless compressor to handle redundancy.







**4. Data Collection & Phase Space Graphing**

* **Approved:**
* * [x] Plot the continuous phase space trajectory mapping $I(t)$ against $P(t)$.




* **Reason:** Swarms do not need constant information (e.g., open space requires less information than a narrow corridor). A time-variant curve shows exactly how dynamic information throttling dictates real-time performance.







## Phase 2: Scaling Complexity (Future Levels)

**5. Level 2: Dynamic Interception**

* **Approved:**
* * [x] Shift to SISL `pursuit` or MPE `simple_tag` using a Homogeneous Dense Swarm.




* **Reason:** Moving targets demand continuous velocity updates, testing medium $I(t)$ demands. The higher density forces agents into confined spaces, pushing the critical performance threshold further to the right on the X-axis (requiring higher bandwidth to prevent gridlock).







**6. Level 3: Asymmetric / Heterogeneous Swarms**

* **Approved:**
* * [x] Shift to MPE `simple_speaker_listener` or SISL `waterworld` using specialized roles.




* **Reason:** Tests high $I(t)$ demands. Agents must encode discrete categorical role bits (e.g., Scout vs. Actuator) alongside high-resolution spatial instructions, drastically increasing the minimum baseline entropy required to coordinate.







**7. Phase 3: 3D Physics & Real-World Porting**

* **Approved:**
* * [x] Maintain the universal bit-rate metric and empirical normalizer framework.




* **Reason:** The framework is designed to be task- and architecture-agnostic, meaning the same Oracle/Blind normalization formulas will seamlessly adapt when applied to heavier 3D engines.






* **TBD:**
* * [ ] Port logic into ROS 2 Jazzy nodes and Gazebo.


* **The Dilemma:** Moving from abstract 2D kinematics to continuous 3D physics introduces real-world mechanical constraints (inertia, motor torque limitations). We must decide how to keep these physical constraints from muddying the $I(t)$ threshold results when scaling up.
