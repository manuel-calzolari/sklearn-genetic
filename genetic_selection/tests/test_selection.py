import random

import numpy as np
import pytest
from sklearn import datasets, linear_model
from genetic_selection import GeneticSelectionCV


@pytest.fixture
def data():
    iris = datasets.load_iris()
    E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
    X = np.hstack((iris.data, E))
    y = iris.target
    return X, y


def test_genetic_selection(data):
    random.seed(42)
    np.random.seed(42)
    X = data[0]
    y = data[1]
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
    np.testing.assert_array_equal(
        selector.support_,
        np.array(
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        ),
    )
