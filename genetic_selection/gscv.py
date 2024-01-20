# sklearn-genetic - Genetic feature selection module for scikit-learn
# Copyright (C) 2016-2024  Manuel Calzolari
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Genetic algorithm for feature selection"""

import numbers
import multiprocess
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import available_if
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.metrics import check_scoring
from sklearn.feature_selection import SelectorMixin
from sklearn.utils._joblib import cpu_count
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)


def _eaFunction(population, toolbox, cxpb, mutpb, ngen, ngen_no_change=None, stats=None, halloffame=None, verbose=0):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("The 'halloffame' parameter should not be None.")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    wait = 0
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Add the best back to population
        offspring.extend(halloffame.items)

        # Get the previous best individual before updating the hall of fame
        prev_best = halloffame[0]

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # If the new best individual is the same as the previous best individual,
        # increment a counter, otherwise reset the counter
        if halloffame[0] == prev_best:
            wait += 1
        else:
            wait = 0

        # If the counter reached the termination criteria, stop the optimization
        if ngen_no_change is not None and wait >= ngen_no_change:
            break

    return population, logbook


def _createIndividual(icls, n, max_features, min_features):
    n_features = np.random.randint(min_features, max_features + 1)
    genome = ([1] * n_features) + ([0] * (n - n_features))
    np.random.shuffle(genome)
    return icls(genome)


def _evalFunction(
    individual, estimator, X, y, groups, cv, scorer, fit_params, max_features, min_features, caching, scores_cache={}
):
    individual_sum = np.sum(individual, axis=0)
    if individual_sum < min_features or individual_sum > max_features:
        return -10000, individual_sum, 10000
    individual_tuple = tuple(individual)
    if caching and individual_tuple in scores_cache:
        return scores_cache[individual_tuple][0], individual_sum, scores_cache[individual_tuple][1]
    X_selected = X[:, np.array(individual, dtype=bool)]
    scores = cross_val_score(
        estimator=estimator, X=X_selected, y=y, groups=groups, scoring=scorer, cv=cv, fit_params=fit_params
    )
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    if caching:
        scores_cache[individual_tuple] = [scores_mean, scores_std]
    return scores_mean, individual_sum, scores_std


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the first fitted estimator if available, otherwise we
    check the unfitted estimator.
    """
    return lambda self: (
        hasattr(self.estimator_, attr) if hasattr(self, "estimator_") else hasattr(self.estimator, attr)
    )


class GeneticSelectionCV(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature selection with genetic algorithm.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    max_features : int or None, optional
        The maximum number of features selected.

    min_features : int or None, optional
        The minimum number of features selected.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

    n_population : int, default=300
        Number of population for the genetic algorithm.

    crossover_proba : float, default=0.5
        Probability of crossover for the genetic algorithm.

    mutation_proba : float, default=0.2
        Probability of mutation for the genetic algorithm.

    n_generations : int, default=40
        Number of generations for the genetic algorithm.

    crossover_independent_proba : float, default=0.1
        Independent probability for each attribute to be exchanged, for the genetic algorithm.

    mutation_independent_proba : float, default=0.05
        Independent probability for each attribute to be mutated, for the genetic algorithm.

    tournament_size : int, default=3
        Tournament size for the genetic algorithm.

    n_gen_no_change : int, default None
        If set to a number, it will terminate optimization when best individual is not
        changing in all of the previous ``n_gen_no_change`` number of generations.

    caching : boolean, default=False
        If True, scores of the genetic algorithm are cached.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    generation_scores_ : array of shape [n_generations]
        The maximum cross-validation score for each generation.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    An example showing genetic feature selection.

    >>> import numpy as np
    >>> from sklearn import datasets, linear_model
    >>> from genetic_selection import GeneticSelectionCV
    >>> iris = datasets.load_iris()
    >>> E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
    >>> X = np.hstack((iris.data, E))
    >>> y = iris.target
    >>> estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
    >>> selector = GeneticSelectionCV(estimator, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True  True  True  True False False False False False False False False
           False False False False False False False False False False False False], dtype=bool)
    """

    def __init__(
        self,
        estimator,
        cv=None,
        scoring=None,
        fit_params=None,
        max_features=None,
        min_features=None,
        verbose=0,
        n_jobs=1,
        n_population=300,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=40,
        crossover_independent_proba=0.1,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=None,
        caching=False,
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.fit_params = fit_params
        self.max_features = max_features
        self.min_features = min_features
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_population = n_population
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.n_generations = n_generations
        self.crossover_independent_proba = crossover_independent_proba
        self.mutation_independent_proba = mutation_independent_proba
        self.tournament_size = tournament_size
        self.n_gen_no_change = n_gen_no_change
        self.caching = caching
        self.scores_cache = {}

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y, groups=None):
        """Fit the GeneticSelectionCV model and the underlying estimator on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        groups : array-like, shape = [n_samples], optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" `cv`
            instance (e.g., `GroupKFold`).
        """
        return self._fit(X, y, groups)

    def _fit(self, X, y, groups=None):
        X, y = check_X_y(X, y, "csr")
        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral):
                raise TypeError(
                    "'max_features' should be an integer between 1 and {} features."
                    " Got {!r} instead.".format(n_features, self.max_features)
                )
            elif self.max_features < 1 or self.max_features > n_features:
                raise ValueError(
                    "'max_features' should be between 1 and {} features."
                    " Got {} instead.".format(n_features, self.max_features)
                )
            max_features = self.max_features
        else:
            max_features = n_features

        if self.min_features is not None:
            if not isinstance(self.min_features, numbers.Integral):
                raise TypeError(
                    "'min_features' should be an integer between 1 and {} features."
                    " Got {!r} instead.".format(n_features, self.min_features)
                )
            elif self.min_features < 1 or self.min_features > n_features:
                raise ValueError(
                    "'min_features' should be between 1 and {} features."
                    " Got {} instead.".format(n_features, self.min_features)
                )
            min_features = self.min_features
        else:
            min_features = 1

        if max_features < min_features:
            max_features = min_features

        if not isinstance(self.n_gen_no_change, (numbers.Integral, np.integer, type(None))):
            raise ValueError(
                "'n_gen_no_change' should either be None or an integer." " {} was passed.".format(self.n_gen_no_change)
            )

        estimator = clone(self.estimator)

        # Genetic Algorithm
        toolbox = base.Toolbox()

        toolbox.register(
            "individual",
            _createIndividual,
            creator.Individual,
            n=n_features,
            max_features=max_features,
            min_features=min_features,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register(
            "evaluate",
            _evalFunction,
            estimator=estimator,
            X=X,
            y=y,
            groups=groups,
            cv=cv,
            scorer=scorer,
            fit_params=self.fit_params,
            max_features=max_features,
            min_features=min_features,
            caching=self.caching,
            scores_cache=self.scores_cache,
        )
        toolbox.register("mate", tools.cxUniform, indpb=self.crossover_independent_proba)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_independent_proba)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        if self.n_jobs == 0:
            raise ValueError("n_jobs == 0 has no meaning.")
        elif self.n_jobs > 1:
            pool = multiprocess.Pool(processes=self.n_jobs)
            toolbox.register("map", pool.map)
        elif self.n_jobs < 0:
            pool = multiprocess.Pool(processes=max(cpu_count() + 1 + self.n_jobs, 1))
            toolbox.register("map", pool.map)

        pop = toolbox.population(n=self.n_population)
        hof = tools.HallOfFame(1, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        if self.verbose > 0:
            print("Selecting features with genetic algorithm.")

        with np.printoptions(precision=6, suppress=True, sign=" "):
            _, log = _eaFunction(
                pop,
                toolbox,
                cxpb=self.crossover_proba,
                mutpb=self.mutation_proba,
                ngen=self.n_generations,
                ngen_no_change=self.n_gen_no_change,
                stats=stats,
                halloffame=hof,
                verbose=self.verbose,
            )
        if self.n_jobs != 1:
            pool.close()
            pool.join()

        # Set final attributes
        support_ = np.array(hof, dtype=bool)[0]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, support_], y)

        self.generation_scores_ = np.array([score for score, _, _ in log.select("max")])
        self.n_features_ = support_.sum()
        self.support_ = support_

        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Reduce X to the selected features and then predict using the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @available_if(_estimator_has("score"))
    def score(self, X, y):
        """Reduce X to the selected features and return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))
