# Markowitz Reference Implementation

This repository accompanies the paper \cite{X}.
It contains a reference implementation of the Markowitz portfolio optimization
problem and the data used in the paper. Please note that the tickers of the
stocks have been obfuscated to comply with the data provider's terms of use.

## Experiments

Please run all experiments using

```bash
make experiments
```

This first replicates the virtual environment defined in 'requirements.txt'
locally and then runs the experiments defined in 'experiments.py'.

## Reproducibility

The main packages used are specified in 'requirements.txt', with a frozen
version of all packages and their sub-dependencies in 'requirements_frozen.txt'.
We used Python 3.10.13 to run the experiments.
