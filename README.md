# sklearn-genetic

**sklearn-genetic** is a genetic feature selection module for scikit-learn.

Genetic algorithms mimic the process of natural selection to search for optimal values of a function.

## Installation

### Dependencies

sklearn-genetic requires:

* Python (>= 3.6)
* scikit-learn (>= 0.23)
* deap (>= 1.0.2)
* numpy
* multiprocess

### User installation

The easiest way to install sklearn-genetic is using `pip`

```bash
pip install sklearn-genetic
```

or `conda`

```bash
conda install -c conda-forge sklearn-genetic
```

## Example

Noisy (non informative) features are added to the iris data and genetic feature selection is applied.

```python
import random
import numpy as np
from sklearn import datasets, linear_model
from genetic_selection import GeneticSelectionCV

# When using multiple processes (n_jobs != 1), protect the entry point of the program if necessary
if __name__ == "__main__":

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    iris = datasets.load_iris()

    # Some noisy data not correlated
    E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

    X = np.hstack((iris.data, E))
    y = iris.target

    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

    selector = GeneticSelectionCV(
        estimator,
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
        n_jobs=-1,
    )
    selector = selector.fit(X, y)

    print(selector.support_)

```

## See also

* [shapicant](https://github.com/manuel-calzolari/shapicant), a feature selection package based on SHAP and target permutation, for pandas and Spark
