# Stochastic Demographic Simulator — Portfolio Showcase

Concise, production-oriented description: a custom-built Discrete Event Simulation (DES) + Agent-Based Model (ABM) implemented in Python to study long-term demographic dynamics. Designed and engineered to demonstrate advanced skills in stochastic modelling, algorithmic optimization, and software architecture.

**Highlights (what this project shows off)**

- Applied probability & simulation: inverse-transform sampling, Bernoulli trials, and hazard-rate modeling.
- Algorithmic design: efficient event scheduling (min-heap priority queue) and optimized pairing logic for matchmaking at scale.
- Statistical rigor: reproducible Monte Carlo pipeline, aggregation into confidence intervals, and interactive visualization via Streamlit.
- Engineering practices: modular architecture, unit tests, and dependency management.

**Tech stack**

- Python 3.10+ (typed where useful)
- NumPy, pandas
- Streamlit for dashboarding
- Pytest for core tests
- tqdm for progress and Monte Carlo runs

**Repository structure**

See the main components:

- [main.py](main.py) — Interactive entry point / REPL
- [run_analysis.py](run_analysis.py) — Orchestrates Monte Carlo runs and exports statistics
- [dashboard.py](dashboard.py) — Streamlit dashboard for visualization
- [engine/simulator.py](engine/simulator.py) — Core DES engine and event loop
- [models/person.py](models/person.py) — Agent data model
- [utils/probability.py](utils/probability.py) — Sampling helpers

Quickstart

1. Clone the repository:

```bash
git clone https://github.com/JosuSC/Stochastic_Demographic_Simulator.git
cd Stochastic_Demographic_Simulator
```

2. (Recommended) Create and activate a virtual environment, then install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

3. Run a single interactive simulation (REPL):

```bash
python main.py
```

4. Run the statistical pipeline (default: 100 Monte Carlo runs):

```bash
python run_analysis.py
```

Outputs

- `simulation_statistics.csv` — yearly aggregated metrics (mean, std, CI)
- `simulation_runs.csv` — raw per-run yearly records for deeper analysis

5. Launch the dashboard:

```bash
streamlit run dashboard.py
```

Why include this in a portfolio

- Demonstrates building a non-trivial simulation engine from first principles (no SimPy). 
- Shows ability to design reproducible experiments and present results with interpretable statistical summaries.
- Highlights competency in performance-aware Python coding and end-to-end analysis.

Contributing

Contributions and bug reports are welcome. Open an issue or submit a PR with clear motivation and tests.

License & author

This repository is provided for portfolio and academic purposes. Author: Josué Javier Senarega Claro.


