import logging
import time
import json
from pathlib import Path

from sklearn.metrics import log_loss
from hyperopt import hp, pyll

import data_preparation as dp 
import xgboost_modeling as xgm

# Path to the root of the project (where this file lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __name__ == '__main__':

    # create directory for log and models if not existing   
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'models').mkdir(exist_ok=True)

    logging.basicConfig(
        filename=PROJECT_ROOT / 'logs' / 'model_information.log',    # log file path
        filemode='a',                    # append mode
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # get and prepare match data from 2003-2023
    tennis_match_data = dp.TennisMatchDataset(range(2003,2024))
    tennis_match_data.process()

    # set up XGB Modeling Config object
    xgb_tuning_configuration = xgm.XGBTuningConfig(
        data = tennis_match_data,
        early_stopping_rounds = 50,
        metric = log_loss,
        objective = 'binary:logistic',
        eval_metric = 'logloss',
        num_boost_round = 2000,
        final_seed = 333 # only used in final models
    )

    # define distributions to search over for hyperparameter tuning
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

    # run search over full space for 1000 rounds and get model
    start_time = time.time()
    params_full_search = xgm.run_tuning_round(
        modeling_config = xgb_tuning_configuration,
        hyperparam_range = total_search_space,
        max_evals = 1000,
        random_seed = 123
    )
    end_time = time.time()
    elapsed_time = end_time-start_time
    logging.info(f'1000 rounds of hyperparameter tuning took {elapsed_time} seconds')
    logging.info(
        'Parameters found after 1000 rounds: %s',
        json.dumps(params_full_search,indent=2)
    )
    model_full_search = xgm.get_trained_xgboost(
        modeling_config = xgb_tuning_configuration,
        hyperparams = params_full_search,
        is_from_fmin = True
    )
    model_full_search.save_model(PROJECT_ROOT / 'models' / 'model_full_search.json')


    # run 20 trials at 25 eval rounds per trial to then extract narrower space
    start_time = time.time()
    trial_results = xgm.run_xgboost_trials(
        modeling_config = xgb_tuning_configuration,
        search_space = total_search_space,
        random_seed = 124,
        num_trials = 20,
        rounds_per_trial = 25
    )
    end_time = time.time()
    elapsed_time = end_time-start_time
    logging.info(f"500 rounds of hyperparameter trials took {elapsed_time} seconds")
    logging.info(
        'Parameters found after 500 trial rounds: %s',
        json.dumps(trial_results,indent=2)
    )

    # use trial results to get models trained on narrower space using percentiles
    percentiles = {
        '10_90': (10,90),
        '20_80': (20,80),
        '30_70': (30,70),
        '40_60': (40,60)
    }
    for key,value in percentiles.items():
        print(f'Starting work on percentile range {key}')
        start_time = time.time()
        percentile_params = xgm.run_tuning_percentile_range(
            modeling_config = xgb_tuning_configuration,
            trial_results = trial_results,
            percentile_range = value,
            max_evals = 200,
            random_seed = list(percentiles).index(key)+125
        )
        end_time = time.time()
        elapsed_time = end_time-start_time
        logging.info(f"200 rounds of hyperparameter trials for percentile {key} took {elapsed_time} seconds")
        logging.info(
            'Parameters found after 200 rounds for percentile %s : %s',
            key,
            json.dumps(percentile_params,indent=2)
        )
        percentile_model = xgm.get_trained_xgboost(
            modeling_config = xgb_tuning_configuration,
            hyperparams = percentile_params,
            is_from_fmin = True
        )
        percentile_model.save_model(PROJECT_ROOT / 'models' / f'model_{key}.json')
    


    # run search over full space for 700 rounds and get model
    start_time = time.time()
    params_shorter_wide_search = xgm.run_tuning_round(
        modeling_config = xgb_tuning_configuration,
        hyperparam_range = total_search_space,
        max_evals = 700,
        random_seed = 129
    )
    end_time = time.time()
    elapsed_time = end_time-start_time
    logging.info(f"700 rounds of hyperparameter tuning took {elapsed_time} seconds")
    logging.info(
        'Parameters found after 700 rounds: %s',
        json.dumps(params_shorter_wide_search,indent=2)
    )
    model_shorter_wide_search = xgm.get_trained_xgboost(
        modeling_config = xgb_tuning_configuration,
        hyperparams = params_shorter_wide_search,
        is_from_fmin = True
    )
    model_shorter_wide_search.save_model(PROJECT_ROOT / 'models' / 'model_shorter_wider_search.json')
