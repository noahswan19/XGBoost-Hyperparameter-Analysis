from typing import Dict, Any, Callable
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

import xgboost as xgb
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
from sklearn.metrics import log_loss, accuracy_score

import data_preparation as dp

@dataclass
class XGBTuningConfig:
    """
    Configuration container for XGBoost hyperparameter tuning experiments.

    Attributes:
        data (dp.TennisMatchDataset): Tennis match data.
        early_stopping_rounds (int): Number of rounds to use for early stopping during training.
        metric (Callable[[np.ndarray, np.ndarray], float]): A scoring function to evaluate
            model performance, typically from `sklearn.metrics` (e.g., `log_loss`, `mean_squared_error`).
        objective (str): String for objective hyperparameter when tuning XGBoost model (e.g. 'binary:logistic').
        eval_metric (str): String for metric hyperparameter when tuning XGBoost model (e.g. 'logloss').
        final_seed (int): Random seed for reproducibility for final models after tuning.
    """
    data: dp.TennisMatchDataset
    early_stopping_rounds: int = 50
    metric: Callable[[np.ndarray,np.ndarray],float] = log_loss
    objective: str = 'binary:logistic'
    eval_metric: str = 'logloss' # just for hyperparameter dictionary that goes in xgb.train
    num_boost_round: int = 2000
    final_seed: int = 333 # seed to be used for getting final models only

def xgboost_objective_function(
        params: Dict[str, Any],
        modeling_config: XGBTuningConfig,
        random_generator: np.random.Generator
)->Dict[str,Any]:
    """
    Objective function for hyperparameter tuning with XGBoost to be minimized
    with hyperopt library.

    Args:
        params (Dict[str, any]): Hyperparameters for XGBoost.
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        random_generator (np.random.Generator): Result of np.random.default_rng that is used to set seed for fmin.

    Returns:
        Dict[str, Any]: Dictionary containing the loss and status of the trial.
    """

    params['max_depth'] = int(params['max_depth'])  # Ensure max_depth is an integer
    seed = random_generator.integers(0,1e9)
    # print(f'Using {seed} as the seed')
    params['seed'] = seed
    params['objective'] = modeling_config.objective
    params['eval_metric'] = modeling_config.eval_metric

    model_variant = xgb.train(
        params,
        dtrain = modeling_config.data.train_xgbmatrix,
        num_boost_round = modeling_config.num_boost_round, # set high to ensure we can find optimal number of rounds with early stopping
        evals = [(modeling_config.data.test_xgbmatrix, 'eval')],
        early_stopping_rounds = modeling_config.early_stopping_rounds,
        verbose_eval = False
    )

    predictions_test = model_variant.predict(modeling_config.data.test_xgbmatrix)
    loss = modeling_config.metric(modeling_config.data.test_xgbmatrix.get_label(), predictions_test)
    # print(f'loss: {loss}')

    # could include model variant in return, but not necessary for now
    return {'loss': loss, 'status': STATUS_OK}

def run_tuning_round(
        modeling_config: XGBTuningConfig,
        hyperparam_range: Dict[str,Any],
        max_evals:int,
        random_seed:int = 123,
        objective_function:Callable[[Any],Dict[str,Any]] = xgboost_objective_function
)->Dict[str,Any]:
    """
    Run fmin with hyperparameter range and given dataset and objective function.

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        hyperparam_range (Dict[str, Any]): Hyperparameters with ranges for the XGBoost model.
        max_evals (int): Number of hyperparameter combinations to test.
        random_seed (int): Random seed for reproducibility, passed to np.random.default_rng.
        objective_function (Callable[[Any],Dict[str,Any]]): Objective function that will be minimized when calling fmin.

    Returns:
        Dict[str, Any]: The hyperparameters from the best iteration of fmin.
    """
    rng = np.random.default_rng(random_seed)

    trials = Trials() # not necessary, but included in case useful in future
    best = fmin(
        fn = lambda x: objective_function(
            x,
            modeling_config = modeling_config,
            random_generator = rng
        ),
        space = hyperparam_range,
        algo = tpe.suggest, # suggests best expected hyperparameter point based on Tree Parzen Estimator (TPE) algorithm
        max_evals=max_evals, # number of hyperparameter combinations to try, limited by computational capacity
        trials=trials,
        rstate=rng
    )

    return best

def run_xgboost_trials(
        modeling_config: XGBTuningConfig,
        search_space: Dict[str, Any],
        random_seed:int = 123,
        num_trials: int = 10,
        rounds_per_trial: int = 25
)->Dict[str,list[Any]]:
    """
    Run XGBoost trials with hyperparameter tuning for shorter runs to narrow search space.

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        search_space (Dict[str, Any]): Hyperparameter search space.
        random_seed (int): Random seed for reproducibility.
        num_trials (int): Number of trials to run.
        rounds_per_trial (int): Number of rounds per trial for fmin; number of parameter combinations to try for each trial.

    Returns:
        Dict[str, list[Any]]: Dictionary containing results of the trials.
    """
    rng = np.random.default_rng(random_seed)  # Set random seed for reproducibility
    # this is picked so that for any run of n trials, the ith call to fmin has the same seed
    # if running 10 trials, each fmin should be seeded the same across calls
    seeds = [rng.integers(0,1e9) for _ in range(num_trials)]
    all_results = []
    for _ in range(num_trials):
        print(f'Running trial {_ + 1} with seed {seeds[_]}')
        best = run_tuning_round(
            modeling_config = modeling_config,
            hyperparam_range=search_space,
            max_evals=rounds_per_trial,
            random_seed = seeds[_]
        )
        all_results.append(best)
    
    consolidated_results = defaultdict(list)
    for result in all_results:
        for key, value in result.items():
            consolidated_results[key].append(value)

    return dict(consolidated_results)
    
def run_tuning_percentile_range(
        modeling_config: XGBTuningConfig,
        trial_results: Dict[str, list[Any]],
        percentile_range: tuple[float,float] = (25, 75),
        max_evals: int = 100,
        random_seed: int = 123
)->Dict[str, Any]:
    """
    Run XGBoost tuning with a specified percentile range based on previous trial results.

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        trial_results (Dict[str, list]): Results from previous trials.
        percentile_range (tuple): Percentile range to narrow the search space (e.g. (10,90) for 10th to 90th percentile).
        max_evals (int): Maximum evaluations for hyperparameter tuning.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Best hyperparameters found within the specified percentile range.
    """
    trial_results_int = deepcopy(trial_results)

    loguniform_keys = ['learning_rate','min_child_weight','gamma',
                       'alpha','lambda']
    
    # Convert loguniform keys to log scale to parameterize the search space
    for key in loguniform_keys:
        trial_results_int[key] = np.log(trial_results_int[key])

    # Calculate percentiles for each hyperparameter to use to create narrower range
    percentiles = {key: np.percentile(value, percentile_range) for key, value in trial_results_int.items()}


    narrowed_search_space = {
        'learning_rate': hp.loguniform('learning_rate', percentiles['learning_rate'][0], percentiles['learning_rate'][1]),
        'max_depth': pyll.scope.int(hp.quniform('max_depth', percentiles['max_depth'][0], percentiles['max_depth'][1], 1)),
        'min_child_weight': hp.loguniform('min_child_weight', percentiles['min_child_weight'][0], percentiles['min_child_weight'][1]),
        'subsample': hp.uniform('subsample', percentiles['subsample'][0], percentiles['subsample'][1]),
        'colsample_bytree': hp.uniform('colsample_bytree', percentiles['colsample_bytree'][0], percentiles['colsample_bytree'][1]),
        'gamma': hp.loguniform('gamma', percentiles['gamma'][0], percentiles['gamma'][1]),
        'alpha': hp.loguniform('alpha', percentiles['alpha'][0], percentiles['alpha'][1]),
        'lambda': hp.loguniform('lambda', percentiles['lambda'][0], percentiles['lambda'][1])
    }

    # Run tuning round using fmin and narrower search space
    best = run_tuning_round(
        modeling_config=modeling_config,
        hyperparam_range=narrowed_search_space,
        max_evals = max_evals,
        random_seed = random_seed
    )
    
    return best

def get_trained_xgboost(
        modeling_config: XGBTuningConfig,
        hyperparams: Dict[str, Any],
        is_from_fmin:bool = True,
)->xgb.Booster:
    """
    Train an XGBoost model with given hyperparameters.

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        hyperparams (Dict[str, Any]): Hyperparameters for the XGBoost model, optionally the output of fmin.
        is_from_fmin (bool): Weather the dictionary of hyperparameters was the result of an fmin tuning result.

    Returns:
        xgboost.Booster: Trained XGBoost model.
    """

    if(is_from_fmin):
        hyperparams['eval_metric'] = modeling_config.eval_metric
        hyperparams['objective'] = modeling_config.objective
        hyperparams['seed'] = modeling_config.final_seed
        if 'max_depth' in hyperparams.keys():
            hyperparams['max_depth'] = int(hyperparams['max_depth']) # ensure max_depth is an integer
    
    model = xgb.train(
        params = hyperparams,
        dtrain = modeling_config.data.train_xgbmatrix,
        evals = [(modeling_config.data.test_xgbmatrix,'eval')],
        num_boost_round = modeling_config.num_boost_round,
        early_stopping_rounds = modeling_config.early_stopping_rounds
    )

    return model

def get_default_hyperparameter_xgboost(
        modeling_config: XGBTuningConfig
) -> xgb.Booster:
    """
    Get XGBoost tuned model trained on dataset, wrapper of get_trained_xgboost

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.

    Returns:
        xgboost.Booster: Trained XGBoost model with default hyperparameters.
    """
    default_params = {
        'objective': modeling_config.objective,
        'seed': modeling_config.final_seed
    }

    default_model = get_trained_xgboost(
        modeling_config=modeling_config,
        hyperparams=default_params,
        is_from_fmin=False
    )

    return default_model

def compare_xgboost_performance(
        data:dp.TennisMatchDataset,
        models:Dict[str,xgb.Booster],
        metric:Callable[[np.ndarray,np.ndarray],float] = log_loss
)->Dict[str,Any]:
    """
    Compare trained XGBoost models on provided dataset using given metric.

    Args:
        data (dp.TennisMatchDataset): Dataset containing testing data to evaluate the models on.
        models (Dict[str,xgb.Booster]): Dictionary with keys representing labels for each model and values representing the model.
        metric (Callable): Function to use to evaluate the models.

    Returns:
        Dict[str,Any]: Dictionary with names of models and numeric values corresponding to results of evaluation.
    """

    preds_dict = {
        k:v.predict(data.test_xgbmatrix) for k,v in models.items()
    }

    # need to round for accuracy_score, more logic may be needed for other metrics
    if metric == accuracy_score:
        preds_dict = {
            k:np.round(v) for k,v in preds_dict.items()
        }

    final_dict = {
        k:metric(
            data.test_xgbmatrix.get_label(),
            v
        ) for k,v in preds_dict.items()
    }
    
    return final_dict
