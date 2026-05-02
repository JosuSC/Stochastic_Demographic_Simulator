# 🌍 Stochastic Demographic Simulator

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Paradigm](https://img.shields.io/badge/paradigm-Agent--Based_Modeling-orange.svg)
![Type](https://img.shields.io/badge/type-Discrete_Event_Simulation-success.svg)

An interactive, **Agent-Based Model (ABM)** and **Discrete Event Simulator (DES)** written in Python. This engine models the demographic evolution of a localized population over a century (100 years). It leverages stochastic probabilities, mathematical distributions, and object-oriented programming to simulate the highly complex, interconnected life cycles of individual agents within a macroscopic environment.

This project was built to demonstrate advanced proficiency in applied mathematics, algorithmic optimization, and system architecture.

## 🚀 Key Features

* **Agent-Based Modeling (ABM):** Each individual is tracked as an independent entity (`Person`) via Object-Oriented design, retaining its own unique memory state, partner references, age, offspring count, and grief periods.
* **Deterministic & Stochastic Engine:** Powered by a custom monthly-tick simulation engine, computing probabilities concurrently for thousands of interacting agents.
* **Complex Matchmaking Algorithm:** Connects single agents dynamically based on age-difference matrices and conditional desire probabilities, achieving $O(N)$ efficiency.
* **Real-World Demographic Mechanics:** Accurately models birth rates, multiple-birth probabilities, and mortality rates utilizing Uniform and Exponential distributions.
* **Macro-World Events:** Integrates environmental randomness such as Global Pandemics, Economic Crises, Medical Breakthroughs, and Baby Booms that scale dynamic variables at runtime.
* **Interactive CLI / REPL:** Features an immersive command-line interface that allows the user to step through the simulation year-by-year, decade-by-decade, while printing real-time analytical logs.

## 📐 Mathematical Foundation
The engine relies on a strict theoretical foundation to prevent mathematical anomalies (e.g., immediate population collapse):

1. **Probability Normalization:** Original annual rates are mathematically projected into monthly temporal steps using standard complement functions ($P_{monthly} = 1 - \sqrt[12]{1 - P_{annual}}$) to accurately frame Bernoulli trials.
2. **Exponential Grief Modeling:** Upon a breakup or the death of a partner, agents enter an emotional recovery period modeled via an Exponential distribution where the scale parameter $\lambda$ depends on their biological age.

## 📂 Architecture & Project Structure

The project relies on a modular, decoupled architecture, entirely independent of external simulation libraries like *SimPy*, proving the capability to build low-level engines from scratch.

```text
/project_root
├── main.py                 # Interactive CLI Entry Point & REPL
├── models/
│   └── person.py           # POJO / Data Class representing an Agent's state
├── engine/
│   └── simulator.py        # Core DES Engine, Tick Processor, and Matchmaker
├── utils/
│   └── probability.py      # Abstracted Stochastic wrappers (Uniform, Exponential, Bernoulli)
└── tests/
    └── test_simulation.py  # Pytest suite
```

## ⚙️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Stochastic_Demographic_Simulator.git
   cd Stochastic_Demographic_Simulator
   ```

2. **Run the Interactive Simulation**
   No external UI dependencies required. Just pure Python runtime:
   ```bash
   python main.py
   ```

3. **CLI Commands**
   Once inside the REPL loop, you can instruct the engine with:
   * `n` or `1` : Advance the simulation by exactly 1 year.
   * `[Any Integer]` : Advance the simulation by $X$ years recursively.
   * `q` or `exit` : Abort the current timeline and trigger the Final Demographic Report.

## 📈 Future Enhancements
- **Spatial Grid:** Incorporate a 2D Cartesian grid or graph network to apply geographic constraints to matchmaking.
- **Genetic Traits:** Pass inherited metadata (disease resistance, fertility caps) from parents to offspring.
- **Parallel Computing:** Migrate the `tick()` cycle to Python's `multiprocessing` to handle populations exceeding 1,000,000 agents.

---
*Created as an academic showcase of computational simulation logic, software architecture, and stochastic modeling.*