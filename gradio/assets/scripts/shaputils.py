import shap
import warnings
import numpy as np
import pandas as pd
import copy
import joblib

import matplotlib
import matplotlib.pyplot as plt
import gradio as gr

import dill as pickle

max_textboxes = 25

warnings.filterwarnings("ignore")
rf_model_path = "assets/files/rf_model_webserver.joblib"


def main(json_data):
    x_test = np.array(json_data["x_test"])
    colnames = json_data["colnames"]
    
    x_df = pd.DataFrame(x_test,
               columns = colnames)
    x_df = x_df[['edss_score',
            'age_at_visit',
            'revised_debut_age',
            'mono_on_sum',
            'monofocal_sum',
            'multifocal_sum',
            'afferent_non_on_sum',
            'steroid_treatment_sum',
            'is_last_relapse_steroid_treated',
            'is_last_relapse_completely_remitted',
            'age_at_relapse',
            'age_at_debut_relapse']]

    rf = joblib.load(rf_model_path)
    explainer = shap.Explainer(rf, x_df)
    shap_values = explainer(x_df)
    
    bar_plot = copy.deepcopy(get_bar_plot(shap_values))
    beeswarm_plot = copy.deepcopy(get_beeswarm_plot(shap_values))

    
    force_plot_list = []
    for i in range(len(x_test)):
        force_plot_list.append(gr.Plot.update(get_force_plot(shap_values,i,x_df),visible=True))
    
    
    if len(x_test) < max_textboxes:
        for i in range(max_textboxes-len(x_test)):
            force_plot_list.append(gr.Plot.update(visible=False))
    
    
    return ([gr.Plot.update(bar_plot,visible=True),gr.Plot.update(beeswarm_plot,visible=True),json_data] + force_plot_list)


def get_force_plot(shap_values,index,x_df):
    matplotlib.rcParams['axes.spines.right'] = True
    matplotlib.rcParams['axes.spines.top'] = True
    force_fig, ax = plt.subplots(1,1)
    feature_names = ['edss\nscore\n', 'age\nat\nvisit\n', 'mono\non\nsum\n', 'revised\ndebut_age\n',
       'monofocal\nsum\n', 'multifocal\nsum\n', 'afferent\nnon\non\nsum\n',
       'steroid\ntreatment\nsum\n', 'is\nlast\nrelapse\nsteroid\ntreated\n',
       'is\nlast\nrelapse\ncompletely\nremitted\n', 'age\nat\nrelapse\n',
       'age\nat\ndebut\nrelapse\n']
    force_fig = shap.plots.force(shap_values[:,:,1][index, ...],matplotlib=True,show=False,feature_names=feature_names)
    force_fig.set_size_inches(15, 9)
    force_fig.tight_layout()
    return force_fig
    


def get_bar_plot(shap_values):
    matplotlib.rcParams['axes.spines.right'] = True
    matplotlib.rcParams['axes.spines.top'] = True
    bar_fig, ax = plt.subplots(1,1)
    bar_plot = shap.plots.bar(shap_values[:,:,1],show=False)
    bar_fig = copy.deepcopy(plt.gcf())
    bar_fig.set_size_inches(15, 9)
    bar_fig.tight_layout()
    return bar_fig

def get_beeswarm_plot(shap_values):
    matplotlib.rcParams['axes.spines.right'] = True
    matplotlib.rcParams['axes.spines.top'] = True
    beeswarm_fig, ax = plt.subplots(1,1)
    beeswarm_plot = shap.plots.beeswarm(shap_values[:,:,1],show=False)
    beeswarm_fig = copy.deepcopy(plt.gcf())
    beeswarm_fig.set_size_inches(15, 9)
    beeswarm_fig.tight_layout()
    return beeswarm_fig


