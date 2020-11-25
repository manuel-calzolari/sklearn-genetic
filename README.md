# sklearn-genetic

Genetic feature selection module for scikit-learn

Genetic algorithms mimic the process of natural selection to search for optimal values of a function.

## Installation

The easiest way to install sklearn-genetic is using `pip`

```bash
pip install sklearn-genetic
```

or `conda`

```bash
conda install -c conda-forge sklearn-genetic
```

## Requirements

* Python >= 2.7
* scikit-learn >= 0.20.3
* DEAP >= 1.0.2

## Example

```python
from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model

from genetic_selection import GeneticSelectionCV


def main():
    iris = datasets.load_iris()

    # Some noisy data not correlated
    E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

    X = np.hstack((iris.data, E))
    y = iris.target

    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=5,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    print(selector.support_)


if __name__ == "__main__":
    main()

```

## Citing sklearn-genetic

Manuel Calzolari. (2020, October 12). manuel-calzolari/sklearn-genetic: sklearn-genetic 0.3.0 (Version 0.3.0). Zenodo. http://doi.org/10.5281/zenodo.4081754

BibTeX entry:
```
@software{manuel_calzolari_2020_4081754,
  author       = {Manuel Calzolari},
  title        = {{manuel-calzolari/sklearn-genetic: sklearn-genetic 
                   0.3.0}},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.3.0},
  doi          = {10.5281/zenodo.4081754},
  url          = {https://doi.org/10.5281/zenodo.4081754}
}
```
