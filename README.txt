Bayesian Optimisation for Electron Ptychography
==============================================

Overview
--------

This repository provides a Bayesian optimisation framework for PtyREX
electron ptychography reconstruction parameter optimisation using
crystallographic Fourier-space metrics.

The framework is designed for optimisation of graphene
reconstructions and replaces an earlier FRC-based objective with
a physically meaningful crystallographic metric derived from
graphene reciprocal-space reflections.

The optimisation framework is responsible for:

    - generating reconstruction parameter proposals
    - creating reconstruction JSON configurations
    - submitting PtyREX jobs to SLURM
    - monitoring job completion
    - evaluating reconstruction quality
    - updating the Bayesian optimiser
    - tracking campaign state
    - displaying optimisation dashboards

This repository is NOT a standalone reconstruction package.

A working PtyREX installation is required.


Main Features
-------------

- Crystallographic Fourier-space evaluation metric
- Gaussian-process Bayesian optimisation
- Automated campaign management
- SLURM batch submission
- Persistent campaign state and recovery
- Human-supervised automated optimisation loops
- Reconstruction quality dashboard
- Safe stopping and restart
- Convergence detection
- Historical warm-start support


Optimisation Objective
----------------------

The optimiser maximises:

    symmetry_weighted_score

The objective supplied to the Bayesian optimiser is:

    loss = -symmetry_weighted_score

The metric is derived from:

    - radial Fourier peak detection
    - azimuthal peak detection
    - hexagonal peak-family matching
    - local 2D peak fitting
    - fit validation
    - integrated peak intensity
    - peak sharpness
    - symmetry weighting

The primary optimisation target is currently:

    ObjectiveMetric.SYMMETRY


Repository Contents
-------------------

bayes_optimisation_helpers.py

    Trial evaluation framework.
    Computes crystallographic metrics from PtyREX reconstructions.

crystal_fourier_metrics.py

    Crystal-specific Fourier-space analysis and scoring.

ptyrex_bayesian_campaign