# pmcmc
Repository containing generic Particle MCMC implementations for high-dimensional data. 

## Particle Filtering Architecture

This library implements particle filters under the general Feynman–Kac framework, 
separating:

1. Inference mechanics (the PF classes)
2. Probabilistic model structure (the FeynmacKac model)

This decoupling allows a single particle-filtering engine to support:

- bootstrap filters
- guided filters
- optimal proposals
- tempered filters
- conditional SMC (Particle Gibbs / PGAS)
- flow-based proposals
- custom MCMC rejuvenation
- and more.


## 1. Model Defines the Proposal — Not the Filter

Each model must implement:

    p0(key, N)               # initial state sampler
    pt(key, x_prev, t)       # proposal kernel q(x_t | x_{t-1}, y_t)
    log_pt(t, x_t, x_prev)   # log density of the proposal q
    log_g(t, x_t, x_prev, y_t)  # emission likelihood (potential)

The PF classes never hard-code a proposal.

Therefore:

- If pt/log_pt correspond to the transition prior, this becomes a Bootstrap PF.
- If pt/log_pt implement a guided or optimal proposal, this becomes a Guided PF.
- If pt/log_pt come from a learned/flow-based proposal, this becomes a Flow-Guided PF.

The algorithm does **not** change — only the model does.


## 2. Filter Classes Are Pure Algorithms

The PF classes implement different SMC recursion patterns:

PF (standard SMC)
    resample → propagate → weight

ConditionalPF (CSMC)
    fixes a reference trajectory and forces particle 0 to follow it
    used in Particle Gibbs and PGAS

BaseTemperedPF
    implements adaptive tempering:
        - builds a temperature schedule β
        - uses bisection to maintain an ESS threshold
        - includes optional MH rejuvenation

TemperedPF
    same recursion as PF but wrapped in the tempering loop


## 3. Why the Architecture Matters

This design provides:

- one inference engine for any state-space model
- plug-and-play proposal distributions
- clean separation of concerns
- rapid experimentation with proposal families
- minimal code duplication

To create a new variant of the PF algorithm: override t0() and step().
To define a new model: implement p0, pt, log_pt, log_g.


With help from: https://github.com/nchopin/particles/blob/master/README.md

                BaseParticleFilter
                     /        \
                    /          \
                PF (standard)   BaseTemperedPF
                   |                 |
           ConditionalPF        TemperedPF
