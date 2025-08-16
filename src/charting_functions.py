from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from xgboost import Booster

def get_model_from_file(file_name:str)->Booster:
    """
    Take a file name for model parameters and return an XGBoost model with those parameters.

    Args:
        file_name (str): File name with stored parameter information.

    Returns:
        xgboost.Booster: Trained xgboost.Booster with parameters from file.
    """
    trained_model = Booster()
    trained_model.load_model(file_name)

    return trained_model


def clean_model_name(model_name:str)->str:
    """
    Converts name of model from file into cleaned name for charting.

    Args:
        model_name (str): Raw model name from files to be converted for charts.

    Returns:
        str: Cleaned model name for charting.
    """
    model_name = model_name.removeprefix('model_')
    if model_name.endswith('0'):
        final_name = model_name.replace('_','-') + ' Percentile'
    else:
        final_name = model_name.replace('_',' ').title()
    
    final_name = final_name.replace(' Wider','')
    
    return final_name

def plot_bar_chart(
        x: list[str],
        y: list[float],
        bar_labels:bool = True,
        percent_labels:bool = False,
        order_bars:bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ylim: Optional[tuple[float,float]] = None
)->plt.Figure:
    """
    Creates and returns a bar chart using matplotlib.pyplot.

    Args:
        x (list): Categories for the x-axis.
        y (list): Values for the y-axis.
        bar_labels (bool): Indicator to add bar labels above each bar.
        percent_labels (bool): Indicator whether to convert bar labels to percentages.
        order_bars (bool): Indicator to order bars based on aesthetic ordering.
        title (str, optional): Title of the chart.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        ylim (tuple[float,float], optional): Tuple indicating upper and lower y-limits.

    Returns:
        matplotlib.pyplot.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots()
    x = [clean_model_name(i) for i in x] # clean model names
    if order_bars:
        x_ordered = [
            'Full Search',
            '10-90 Percentile',
            '20-80 Percentile',
            '30-70 Percentile',
            '40-60 Percentile',
            'Shorter Search'
        ]
        y_ordered = [y[x.index(i)] for i in x_ordered] # order y-values in same way as x
        x = x_ordered
        y = y_ordered
    
    bars = ax.bar(x, y,color = "#7a5195")
    ax.set_xticklabels(x, rotation=45,ha = 'right',fontname = 'Times New Roman')
    if(bar_labels):
        if percent_labels:
            labels = [f'{v*100:.3f}%' for v in y] # format labels as percentages
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals = 1)) # change axis ticks to be percentages
        else:
            labels = [f'{v:.5f}' for v in y]
        ax.bar_label(bars,padding = 3,labels = labels,font = 'Times New Roman',fontweight = 'bold')
    if title:
        ax.set_title(title,fontname = 'Times New Roman')
    if xlabel:
        ax.set_xlabel(xlabel,fontname = 'Times New Roman')
    if ylabel:
        ax.set_ylabel(ylabel,fontname = 'Times New Roman')
    if ylim:
        ax.set_ylim(*ylim)

    for label in ax.get_xticklabels():
        label.set_fontproperties('Times New Roman')

    for label in ax.get_yticklabels():
        label.set_fontproperties('Times New Roman')

    return fig

