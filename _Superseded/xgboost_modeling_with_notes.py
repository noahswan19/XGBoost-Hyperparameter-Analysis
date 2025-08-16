import data_preparation_with_notes as dp
import xgboost as xgb
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
from typing import Dict, Any, Callable
import random
from collections import defaultdict
from dataclasses import dataclass



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


# stepwise pattern from that ch 13 of the book and github page
# how much does hyperparameter tuning help?
# something that is a function of the range provided in presentation
# need some way of narrowing the search space somewhat empirically
# things to record: time taken, best and median AUC, best and median logloss, accuracy
# Steps:
# - baseline model tuning over full range for 500 rounds to get AUC, logloss, and accuracy
# - run a version on just 25 rounds for wide space 10 times
# - use these 10 runs to get narrower search space
# - run a version on narrow range for 100 rounds to get AUC, logloss, and accuracy
# - run a version for 350 rounds to see if actual improvement

# Functions to implement/to-dos:
# - wide search function -> DONE
# - compare results function -> DONE
# - return model from fmin best tune -> DONE
# - default model return -> DONE
# - compare + charts
# - figure out how hyperopt with tpe works -> DONE ISH
# - put XGBMatrix into data_preparation -> DONE
# - create dataclass for model config -> Done
# - any questions remaining on how code works
#    - Trials() object DONE
#    - fmin() function with multidimensional search space how does it work under the hood
#    - Callable type hints fixing lol
#    - early stopping round parameterize
#    - random seed requirements (consecutive or diff seeds) DONE
#    - picklihng models


my_tuning_config = XGBTuningConfig(
    data = dp.test_tennis_data
)

# loguniform_vals = [
#     pyll.stochastic.sample(hp.loguniform('max_depth', -7, 0))
#     for _ in range(10_000)
# ]
# transformed = np.log(loguniform_vals)

# fig, ax = plt.subplots()   # Create a figure and one subplot (axes)
# ax.hist(loguniform_vals, bins=50, edgecolor='black')  # Plot histogram
# ax.set_title("My Plot")
# ax.set_xlabel("X axis")
# ax.set_ylabel("Y axis")
# plt.show()

total_search_space = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'max_depth': pyll.scope.int(hp.quniform('max_depth', 1, 100,1)),
    'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.loguniform('gamma', -10, 10),
    'alpha': hp.loguniform('alpha', -10, 10),
    'lambda': hp.loguniform('lambda', -10, 10)
}





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
    print(f'Using {seed} as the seed')
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


    return {'loss': loss, 'status': STATUS_OK, 'model': model_variant}


# def_params = xgb.XGBClassifier().fit(dp.test_tennis_data.train_xgbmatrix.get_data(),dp.test_tennis_data.train_xgbmatrix.get_label()).get_params()
# def_params['seed'] = 123

# default_model = xgb.train(
#     params = def_params,
#     dtrain = tennis_train_data,
#     evals = [(tennis_test_data,'eval')],
#     num_boost_round = 2000,
#     early_stopping_rounds = 50
# )

# best['eval_metric'] = 'logloss'
# best['objective'] = 'binary:logistic'
# best['seed'] = 123
# best['max_depth'] = int(best['max_depth'])  # Ensure max_depth is an integer

# baseline_best_model = xgb.train(
#     params = best,
#     dtrain = tennis_train_data,
#     evals = [(tennis_test_data,'eval')],
#     num_boost_round = 2000, 
#     early_stopping_rounds = 50
# )

# default_test = default_model.predict(tennis_test_data)
# baseline_test = baseline_best_model.predict(tennis_test_data)
# print(
#     log_loss(tennis_test_data.get_label(), default_test),
#     log_loss(tennis_test_data.get_label(), baseline_test),
#     accuracy_score(tennis_test_data.get_label(), np.round(default_test)),
#     accuracy_score(tennis_test_data.get_label(), np.round(baseline_test))
# )

# function that takes a range of hyperparameters and returns best set in dictionary
def run_tuning_round(
        modeling_config: XGBTuningConfig,
        hyperparam_range: Dict[str,Any],
        max_evals:int,
        random_seed:int = 123,
        objective_function:Callable[[Any],Dict[str,Any]] = xgboost_objective_function
)->Dict[str,np.float64]:
    """
    Run fmin with hyperparameter range and given dataset and objective function.

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        hyperparam_range (Dict[str, Any]): Hyperparameters with ranges for the XGBoost model.
        max_evals (int): Number of hyperparameter combinations to test.
        random_seed (int): Random seed for reproducibility, passed to np.random.default_rng
        objective_function (Callable): Objective function that will be minimized when calling fmin.

    Returns:
        Dict[str, np.float64]: Trained XGBoost model.
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


baseline_test = run_tuning_round(
    my_tuning_config,
    total_search_space,
    max_evals = 15,
    random_seed = 123
)

# function that returns an actual model from a dictionary of floats
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
        xgb.Booster: Trained XGBoost model.
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
        # data: dp.TennisMatchDataset,
        # random_seed: int = 123
        # early_stopping_rounds: int = 50
) -> xgb.Booster:
    """
    Get XGBoost tuned model trained on dataset, wrapper of get_trained_xgboost

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        # data (dp.TennisMatchDataset): Dataset containing training and testing data.
        # random_seed (int): Random seed for reproducibility.
        # early_stopping_rounds (int): Early stopping rounds for training.

    Returns:
        Dict[str, Any]: Default hyperparameters for XGBoost model.
    """
    default_params = {
        'objective': modeling_config.objective,
        'seed': modeling_config.final_seed
    }

    default_model = get_trained_xgboost(
        modeling_config=modeling_config,
        hyperparams=default_params,
        # random_seed=random_seed,
        # early_stopping_rounds=early_stopping_rounds,
        is_from_fmin=False
    )

    return default_model

baseline_model = get_trained_xgboost(
    modeling_config=my_tuning_config,
    hyperparams=baseline_test,
    is_from_fmin=True
)

default_model = get_default_hyperparameter_xgboost(
    modeling_config=my_tuning_config
)

# can further parameterize if wanted
def run_xgboost_trials(
        modeling_config: XGBTuningConfig,
        search_space: Dict[str, Any],
        random_seed:int = 123,
        num_trials: int = 10,
        rounds_per_trial: int = 25
)->Dict[str,list[np.float64]]:
    """
    Run XGBoost trials with hyperparameter tuning for shorter runs to narrow search space.

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        search_space (Dict[str, Any]): Hyperparameter search space.
        random_seed (int): Random seed for reproducibility.
        num_trials (int): Number of trials to run.
        rounds_per_trial (int): Number of rounds per trial for fmin; number of parameter combinations to try for each trial.

    Returns:
        Dict[str, list[np.float64]]: Dictionary containing results of the trials.
    """
    rng = np.random.default_rng(random_seed)  # Set random seed for reproducibility
    # this is picked so that for any run of n trials, the ith call to fmin has the same seed
    # if running 10 trials, each fmin should be seeded the same across calls
    seeds = [rng.integers(0,1e9) for _ in range(num_trials)]
    all_results = []
    for _ in range(num_trials):
        print(f"Running trial {_ + 1} with seed {seeds[_]}")
        best = run_tuning_round(
            modeling_config = modeling_config,
            hyperparam_range=search_space,
            max_evals=rounds_per_trial,
            random_seed = seeds[_]
        )
        # best = fmin(
        #     fn = lambda x: xgboost_objective_function(
        #         x,
        #         train = xgb.DMatrix(data.X_train.drop('match_id', axis=1), label=data.y_train),
        #         test = xgb.DMatrix(data.X_test.drop('match_id', axis=1), label=data.y_test),
        #         early_stopping_rounds = 50,
        #         metric = log_loss
        #     ),
        #     space = search_space,
        #     algo = tpe.suggest,
        #     max_evals = rounds_per_trial,
        #     rstate=np.random.default_rng(seeds[_])  # Use different seed for each trial
        # )
        all_results.append(best)
    
    consolidated_results = defaultdict(list)
    for result in all_results:
        for key, value in result.items():
            consolidated_results[key].append(value)

    return dict(consolidated_results)
    

test = run_xgboost_trials(
    modeling_config= my_tuning_config,
    search_space= total_search_space,
    num_trials = 5,
    rounds_per_trial = 3
)

def run_tuning_percentile_range(
        modeling_config: XGBTuningConfig,
        trial_results: Dict[str, list[np.float64]],
        percentile_range: tuple[float,float] = (25, 75),
        max_evals: int = 100,
        random_seed: int = 123
)->Dict[str, np.float64]:
    """
    Run XGBoost tuning with a specified percentile range based on previous trial results.

    Args:
        modeling_config (XGBTuningConfig): Model tuning config with data, early stopping rounds, and metric.
        trial_results (Dict[str, list]): Results from previous trials.
        percentile_range (tuple): Percentile range to narrow the search space.
        max_evals (int): Maximum evaluations for hyperparameter tuning.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Best hyperparameters found within the specified percentile range.
    """

    loguniform_keys = ['learning_rate','min_child_weight','gamma',
                       'alpha','lambda']
    
    # Convert loguniform keys to log scale to parameterize the search space
    for key in loguniform_keys:
        trial_results[key] = np.log(trial_results[key])

    # Calculate percentiles for each hyperparameter to use to create narrower range
    percentiles = {key: np.percentile(value, percentile_range) for key, value in trial_results.items()}


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

test_2 = run_tuning_percentile_range(
    modeling_config=my_tuning_config,
    trial_results = test,
    percentile_range = (10,90),
    max_evals = 10
)




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

baseline_model
default_model


compare_xgboost_performance(
    dp.test_tennis_data,
    models = {
        'baseline': baseline_model,
        'default': default_model
    },
    metric = accuracy_score
)


# Steps
# wide search
# - 1 seed for fmin, different seed every xgboost round, but sequence should be same
# trials
# - 1 seed for each fmin trial, different seed each xgboost but sequence should be same for given fmin
# - want 10 trials to be same each time, so set that in the function but note in comment
# - will need to change random seeds across calls to run_tuning_round


# Final calls that are needed ultimately
# - run_tuning_round wide space
# - run_xgboost_trials with trials (diff seed)
# - run percentile range however many times (diff seed each time)
# - get final models, including default, all same seed
