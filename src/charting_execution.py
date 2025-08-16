import os
from pathlib import Path

from sklearn.metrics import log_loss, accuracy_score

import xgboost_modeling as xgm
import data_preparation as dp
import charting_functions as cf

# Path to the root of the project (where this file lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


if __name__ == '__main__':
    # create directory for charts if not existing
    (PROJECT_ROOT / 'charts').mkdir(exist_ok=True)

    # get and prepare match data from 2003-2023
    tennis_match_data = dp.TennisMatchDataset(range(2003,2024))
    tennis_match_data.process()

    # load in model parameters to compare performance
    model_files = os.listdir(PROJECT_ROOT / 'models')
    models = {
        file.split('.')[0]: cf.get_model_from_file(PROJECT_ROOT / 'models' / file)
        for file in model_files
    }

    # get logloss and accuracy for each model
    model_loglosses = xgm.compare_xgboost_performance(
        data = tennis_match_data,
        models = models,
        metric = log_loss
    )
    model_accuracies = xgm.compare_xgboost_performance(
        data = tennis_match_data,
        models = models,
        metric = accuracy_score
    )

    # logloss plots, zooming in to see effect (although bad practice)
    logloss_zoom_levels = {
        'out':(0,0.2),
        'mid':(0.1,0.19),
        'in':(0.175,0.18)
    }
    for key,ylim_tuple in logloss_zoom_levels.items():
        logloss_plot = cf.plot_bar_chart(
            list(model_loglosses.keys()),
            list(model_loglosses.values()),
            title = 'Log Loss Model Comparison',
            ylabel = 'Log Loss',
            ylim = ylim_tuple
        )
        logloss_plot.savefig(
            PROJECT_ROOT / 'charts' / f'logloss_zoomed_{key}.png',
            dpi = 300,
            bbox_inches = 'tight'
        )
    # accuracy plots
    accuracy_zoom_levels = {
        'out': (0,1),
        'in': (0.92,0.925)
    }
    for key,ylim_tuple in accuracy_zoom_levels.items():
        accuracy_plot = cf.plot_bar_chart(
            list(model_accuracies.keys()),
            list(model_accuracies.values()),
            title = 'Accuracy Model Comparison',
            ylabel = 'Accuracy',
            percent_labels = True,
            ylim = ylim_tuple
        )
        accuracy_plot.savefig(
            PROJECT_ROOT / 'charts' / f'accuracy_zoomed_{key}.png',
            dpi = 300,
            bbox_inches = 'tight'
        )
