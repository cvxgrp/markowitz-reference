# Markowitz Reference Implementation

This repository accompanies the paper [Markowitz Portfolio Construction at Seventy](https://web.stanford.edu/~boyd/papers/markowitz.html).
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

## Citation

If you want to reference our work in your research, please consider using the following BibTeX for the citation:

```BibTeX
@misc{boyd2024markowitz,
      title={Markowitz Portfolio Construction at Seventy},
      author={Stephen Boyd and Kasper Johansson and Ronald Kahn and Philipp Schiele and Thomas Schmelzer},
      year={2024},
      doi = {10.48550/arXiv.2401.05080},
      url = {https://arxiv.org/abs/2401.05080}
}
```
