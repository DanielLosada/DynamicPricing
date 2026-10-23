# Dynamic Pricing with Distributional Constraint Learning

Research implementation of personalized pricing under uncertain demand. A
neural network learns how price and client characteristics affect the demand
distribution. The trained network is embedded in a stochastic optimization
model to choose prices that maximize expected revenue across a group of clients.

This project accompanies Daniel Losada Molina's Master's thesis, *Dynamic
Pricing - Neural network-based distributional constraint learning methodology
for mixed-integer stochastic optimization*, for the Master in Statistics for
Data Science at Universidad Carlos III de Madrid, supervised by Carlos Ruiz
Mora and Carlos Heredia Pimienta.

## Problem and approach

The price offered to a client influences both expected demand and its
uncertainty. The project applies **Distributional Constraint Learning (DCL)**
to learn that relationship and use it directly when optimizing prices.

The workflow in [opt.ipynb](MyImplementation/opt.ipynb) has five stages:

1. **Generate synthetic demand.** Each client has two contextual features and
   observations at different prices. Mean demand decreases through a sigmoid
   function, while standard deviation varies with price. Gaussian samples are
   clipped at zero to produce nonnegative observations.
2. **Learn the conditional distribution.** A PyTorch feedforward network takes
   price and contextual features as inputs and predicts demand mean and standard
   deviation. Training uses Gaussian negative log-likelihood, Adam, and early
   stopping. The client ID groups observations but is not a network input.
3. **Evaluate the fit.** Compare validation NLL with a constant Gaussian baseline,
   inspect standardized residuals using a histogram and Q-Q plot, and run a
   Kolmogorov-Smirnov test. Test-set NLL is evaluated separately.
4. **Embed the trained network.** Export weights and biases and represent hidden
   ReLU activations with big-M constraints and binary variables in Pyomo.
   Standard normal draws generate demand scenarios whose mean and standard
   deviation depend on the prices being optimized.
5. **Optimize and experiment.** Solve for client-specific prices with Gurobi,
   inspect demand and revenue, and study how scenario count and client count
   affect prices and solve time.

## Pricing model

The objective is the average total revenue across equally weighted demand
scenarios. Contextual features remain fixed while the offered prices are
decision variables. The formulation supports continuous demand or an integer
approximation constrained to lie within 0.5 units of each sampled demand.

Optional business constraints require the average offered price to be at least
2% above its historical average and cap each client's increase at 5% above that
client's historical average. A general price bound is also enforced. The
average-price requirement couples decisions across clients, so optimizing each
client independently does not generally solve the joint problem.

Although the hidden network embedding uses linear constraints and binary
variables, revenue contains products of price and demand. The resulting model
is a nonconvex mixed-integer quadratic optimization problem, solved with
Gurobi's `NonConvex=2` setting.

## Experiments and findings

The case study uses 100 synthetic clients with 100 observations each. The
network has one 32-unit hidden representation and separate mean and standard
deviation outputs. Data are split within each client into 60% training, 20%
validation, and 20% test observations, preserving row order.

The notebook includes:

- Distributional fit diagnostics and comparison with a constant baseline.
- Price stability and solve time as the number of demand scenarios increases.
- Solve time as the number of jointly optimized clients increases.
- Comparisons with and without historical-average price constraints.
- A five-client pricing example with 50 scenarios and integer demand.

The thesis reports a validation Gaussian NLL of **-0.9185**, compared with
**0.5103** for the baseline, and a standardized-residual KS statistic of
**0.0123** with a p-value of **0.9208**. These are reported case-study results;
a high p-value alone does not establish that predictions are calibrated for
every client or price.

In the reported scenario experiments, 50 scenarios provided sufficiently
stable prices for the case study. Joint optimization became increasingly
expensive as the number of clients grew, making scalability the main limitation.
Decomposition of the coupled pricing problem is a proposed direction for future
work.

## Repository structure

| Path | Purpose |
| --- | --- |
| [MyImplementation/opt.ipynb](MyImplementation/opt.ipynb) | End-to-end training, pricing, diagnostics, and experiments. |
| [MyImplementation/trainer.py](MyImplementation/trainer.py) | Data splitting and scaling, training, prediction, evaluation, and constraint export. |
| [MyImplementation/distModel.py](MyImplementation/distModel.py) | Distributional neural network with mean and standard deviation outputs. |
| [MyImplementation/utils.py](MyImplementation/utils.py) | Synthetic data generators, dataset helpers, network embedding, and plotting. |
| `MyImplementation/data/` | Synthetic transaction dataset for alternative data experiments. |
| `MyImplementation/imgs/` | Demand and optimization experiment figures. |
| `MyImplementation/residual_diagnostics/` | Standardized-residual histogram and Q-Q plot. |
| `MyImplementation/constraints.csv`, `MyImplementation/best_model.pth` | Saved model artifacts from experiments. |

## Getting started

Use a Python environment with the packages in [requirements.txt](requirements.txt)
and a working Gurobi license. PyTorch uses CUDA when available and otherwise
runs on the CPU.

From the repository root, with your virtual environment activated, run the
following in PowerShell:

```powershell
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force MyImplementation/models
Set-Location MyImplementation
jupyter lab opt.ipynb
```

Run the notebook with its kernel working directory set to `MyImplementation`;
imports, checkpoints, and figure paths are relative to that directory. Train
the network and regenerate its constraints before solving so that the
optimization uses the same model checkpoint and scaling statistics.

The main settings are defined in the notebook: `DATA_BY_ID` selects the synthetic
data setup, `config` controls network training, `n_preds` sets the number of
clients, and `n_sc` sets the number of scenarios. In `distModel`,
`hidden_layers` counts layers after the initial input-to-hidden transformation,
so the default `hidden_layers=0` still produces one 32-unit hidden representation.
The final experiment cells solve multiple optimization problems and can take
substantially longer than the five-client example.

## Scope and limitations

Experiments use synthetic data rather than an operational pricing dataset.
The ordered split imitates historical observations, but the generated data do
not contain a real time series. Demand is modeled conditionally as Gaussian;
optimization scenarios can therefore contain negative demand even though the
training observations are clipped at zero. Fit diagnostics should be interpreted
with these assumptions in mind.

The implementation is intended for research experiments. Solver runtime depends
on network size, client count, scenario count, and pricing constraints. Reported
thesis results describe the documented experimental setup and are not runtime
or accuracy guarantees for other configurations.

## Acknowledgements and source attribution

The methodology builds on Antonio Alcantara and Carlos Ruiz's *A neural
network-based distributional constraint learning methodology for mixed-integer
stochastic optimization* (Expert Systems with Applications, 2023) and its
[DistCL reference implementation](https://github.com/antonioalcantaramata/DistCL/tree/main).

The pricing formulation and client-level experiments apply that methodology to
the thesis case study. Network construction, scaling, training and inference,
constraint extraction, and network embedding include code copied or adapted
from DistCL. Comments beside the relevant definitions identify the original
source and describe modifications.
