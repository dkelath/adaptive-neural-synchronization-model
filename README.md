# Adaptive Neural Synchronization Model

A computational neuroscience project modelling synchronisation in networks of phase-reduced FitzHugh–Nagumo oscillators with adaptive synaptic coupling.

## Overview

This project studies how **metabolic constraints shape synchronisation in neural networks**. I model a network of phase oscillators derived from FitzHugh–Nagumo neurons, where synaptic weights are not fixed but evolve over time according to a plasticity rule derived from a cost functional.

The key idea is to balance two competing pressures:

* **Synchrony benefit**: neurons that oscillate in phase communicate and coordinate more effectively.
* **Metabolic cost**: strong synaptic connections are energetically expensive to maintain.

Using a gradient-descent formulation of this trade-off, I derive an adaptive synaptic plasticity rule and simulate how network connectivity and coherence evolve over time.

## Model

Each neuron is represented by a phase variable $\theta_i(t)$, and the network dynamics are given by

$$
\frac{d\theta_i}{dt} = \omega_i + \sum_{j\neq i} K_{ij}(t)\sin(\theta_j-\theta_i)
$$

where $K_{ij}(t)$ is the synaptic weight from neuron (j) to neuron (i).

Synaptic weights evolve according to

$$
\frac{dK_{ij}}{dt} = \varepsilon \cos(\theta_j-\theta_i) - \gamma K_{ij}
$$

This rule is derived from minimising the cost functional

$$
E(K,\theta) = -\frac12 \sum_{i,j} K_{ij}\cos(\theta_j-\theta_i)
$$
to
$$
    \frac{\lambda}{2}\sum_{i,j} K_{ij}^2
$$

which balances synchronisation against metabolic expenditure.

## What this project does

* derives an adaptive synaptic plasticity rule from a variational cost functional
* simulates networks of coupled phase-reduced FitzHugh–Nagumo oscillators
* tracks the Kuramoto order parameter (R(t)) as a measure of synchrony
* visualises phase trajectories, spike rasters, coupling evolution, and phase distributions
* explores how increasing metabolic cost changes the network from synchronised to clustered to incoherent regimes

## Main findings

In simulations of networks with 50–100 neurons, increasing the metabolic cost parameter ( \lambda = \gamma/\varepsilon ) reduces synchrony and changes the qualitative behaviour of the network:

* **low metabolic cost** → near-perfect synchronisation
* **intermediate metabolic cost** → clustered / anti-phase states
* **high metabolic cost** → incoherence and weak coupling

The cost functional decreases monotonically during simulation, consistent with the interpretation of the plasticity rule as gradient descent on a biologically motivated energy functional.

## Repository structure

* `src/adaptive_network.py` — main implementation of the adaptive oscillator network
* `results/` — generated plots from simulations
* `paper/adaptive_coupled_neurons.pdf` — project paper describing derivation, methods, and discussion

## Example outputs

This repository includes visualisations of:

* phase evolution over time
* Kuramoto order parameter (R(t))
* spike raster plots
* mean synaptic coupling strength
* final phase-difference distributions
* cost functional (E(t))

## Running the code

### 1. Clone the repository

```bash
git clone https://github.com/dkelath/adaptive-neural-synchronization-model.git
cd adaptive-neural-synchronization-model
```

### 2. Create an environment and install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the simulation

```bash
python src/adaptive_network.py
```

## Dependencies

* Python 3
* NumPy
* SciPy
* Matplotlib

## Future directions

Possible extensions include:

* heterogeneous intrinsic frequencies
* sparse or modular network topologies instead of all-to-all coupling
* asymmetric / STDP-like plasticity rules
* noisy dynamics and stochastic synaptic updates
* comparison with more biophysically realistic neuron models

## Author

**Dhruv Kelath**
Mathematics major, University of Melbourne
Interested in computational biology and dynamical systems.
