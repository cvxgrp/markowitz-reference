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

This first installs uv on the fly and then creates and replicates the
environment defined in the script section of the 'experiments.py' file
as described in the [documentation of uv run](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)

If you have uv installed you can bypass make and run directly

```bash
uv run experiments.py
```

The script section is as of today

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mosek==11.0.27",
#   "loguru==0.7.3",
#   "numpy==2.3.2",
#   "pandas[output-formatting]==2.3.1",
#   "matplotlib==3.10.5",
#   "cvxpy-base==1.7.1",
#   "clarabel==0.11.1"
# ]
# ///
```

A large fraction of our experiments have been performed
using [MOSEK](https://www.mosek.com/) as the underlying solver.
We assume a valid license for MOSEK is installed. If not,
you may want to apply for a [Trial License](https://www.mosek.com/try/)

## Citation

If you want to reference our work in your research, please consider using the following BibTeX for the citation:

```BibTeX
@article{boyd2024markowitz,
      title={Markowitz Portfolio Construction at Seventy},
      author={S. Boyd and K. Johansson and R. Kahn and P. Schiele and T. Schmelzer},
      journal={Journal of Portfolio Management},
      volume={50},
      number={8},
      pages={117--160},
      year={2024}
}
```
or for the arXiv version:
```BibTeX
@misc{boyd2024markowitz,
      title={Markowitz Portfolio Construction at Seventy},
      author={Stephen Boyd and Kasper Johansson and Ronald Kahn and Philipp Schiele and Thomas Schmelzer},
      year={2024},
      doi = {10.48550/arXiv.2401.05080},
      url = {https://arxiv.org/abs/2401.05080}
}
```
