import sys
import json
import copy
from datetime import datetime

import gradio as gr

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 18})


import joblib

import numpy as np

import pandas as pd

import nonconformist

import warnings

warnings.filterwarnings("ignore")



icp_model_path = "assets/files/icp_model_webserver.joblib"

    
##### INPUT variable size controller - START

max_textboxes = 25
    
def hide_visit_row(k):
    k = int(k)
    input_values_row = []
    for i in range(max_textboxes):
        if i < k:
            visit_date = gr.Textbox.update(visible=True)
            edss_value = gr.Number.update(visible=True)
        else:
            visit_date = gr.Textbox.update(visible=False,value=-1)
            edss_value = gr.Number.update(visible=False,value=-1)
        
        input_values_row.append(edss_value)
        input_values_row.append(visit_date)
    return (input_values_row)


def hide_relapse_row(k):
    k = int(k)
    input_values_row = []
    for i in range(max_textboxes):
        if i < k:
            visit_relapse_date = gr.Textbox.update(visible=True)
            monofocal_relapse = gr.Dropdown.update(visible=True,value="0")
            steroid_treated = gr.Dropdown.update(visible=True,value="0")
            relapse_remission = gr.Dropdown.update(visible=True,value="0")
            mono_on = gr.Dropdown.update(visible=True,value="0")
            afferent_non_on = gr.Dropdown.update(visible=True,value="0")
        else:
            visit_relapse_date = gr.Textbox.update(visible=False,value=-1)
            monofocal_relapse = gr.Dropdown.update(visible=False,value="-1")
            steroid_treated = gr.Dropdown.update(visible=False,value="-1")
            relapse_remission = gr.Dropdown.update(visible=False,value="-1")
            mono_on = gr.Dropdown.update(visible=False,value="-1")
            afferent_non_on = gr.Dropdown.update(visible=False,value="-1")
            
        input_values_row.append(visit_relapse_date)
        input_values_row.append(monofocal_relapse)
        input_values_row.append(steroid_treated)
        input_values_row.append(relapse_remission)
        input_values_row.append(mono_on)
        input_values_row.append(afferent_non_on)
    return (input_values_row)


##### INPUT variable size controller - END


##### PLOTS - START


def each_point_plot_properties(pvals,test_conformal_output):
    no_pred_label,rr_label,sp_label,multiton_label = True,True,True,True
    for i,cpi in enumerate(test_conformal_output):
        labelling = False
        x_,y_ = pvals[i][0],pvals[i][1]
        if len(cpi) == 0:
            shape = '|'
            if no_pred_label:
                label= "No prediction"
                no_pred_label,labelling = False,True
            color_shape = "red"
            marker_fillstyle = "full"
            markersize=15
        if len(cpi) == 1 and cpi[0] == 0:
            shape = "^"
            if rr_label:
                label= "RR"
                rr_label,labelling = False,True
            color_shape = "olivedrab"
            marker_fillstyle = "none"
            markersize=12
        if len(cpi) == 1 and cpi[0] == 1:
            shape = 'x'
            if sp_label:
                label= "SP"
                sp_label,labelling = False,True
            color_shape = "olivedrab"
            marker_fillstyle = "full"
            markersize=12
        if len(cpi) == 2:
            shape = 's'
            if multiton_label:
                label= "RR | SP"
                multiton_label,labelling = False,True
            color_shape = "brown"
            marker_fillstyle = "full"
            markersize=12
        if labelling:
            plt.plot(x_,y_,shape,color=color_shape,label=label, markersize=markersize,fillstyle=marker_fillstyle)
        else:
            plt.plot(x_,y_,shape,color=color_shape, markersize=markersize,fillstyle=marker_fillstyle)

def p0p1plot_for_slider(json_data):
    return gr.Plot.update(p0p1plot(json_data),visible=True)


def p0p1plot(json_data): #pvals,test_conformal_output,age_at_visit_order):
    pvals = np.array(json_data["test_pval"])
    age_at_visit_order = json_data["age_at_visit_order"]
    test_conformal_output = json_data["test_class"]
    
    matplotlib.rcParams['axes.spines.right'] = True
    matplotlib.rcParams['axes.spines.top'] = True
    fig = plt.figure()
    fig.set_size_inches(15, 9)

    x = pvals[:,:1].squeeze()
    y = pvals[:,1:].squeeze()
    plt.plot(x,y)
    plt.xlabel("RRMS p-value")
    plt.ylabel("SPMS p-value")

    visit_time_baseline_corrected = list(np.array(age_at_visit_order) - min(age_at_visit_order))
    for i,(xi,yi) in enumerate(zip(x,y)):
        plt.text(xi+(xi*0.01), yi+(yi*0.01),round(visit_time_baseline_corrected[i],2), size=15,rotation = 45)

    each_point_plot_properties(pvals,test_conformal_output)
    plt.legend()
    plt.xlim(-0.1, 1)
    plt.ylim(-0.1, 1)
    return fig


def disease_course_plot_for_slider(json_data):
    return gr.Plot.update(disease_course_plot(json_data),visible=True)


def disease_course_plot(json_data):
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False

    test_pvalues = np.array(json_data["test_pval"])
    age_at_visit_order = json_data["age_at_visit_order"]
    test_conformal_output = json_data["test_class"]

    input_list = []
    y_values = []
    for entry in test_conformal_output:
        if len(entry) == 1:
            if entry[0] == 0:
                input_list.append("RR")
                y_values.append(0)
            else:
                input_list.append("SP")
                y_values.append(2)
        if len(entry) == 2:
            input_list.append("RR|SP")
            y_values.append(1)
        if len(entry) == 0:
            input_list.append("no prediction")
            y_values.append(3)

    fig = plt.figure()
    fig.set_size_inches(15, 9)
    plot = plt.plot(y_values)
    x,y = plot[0]._x,plot[0]._y

    plt.xlabel("Years since disease onset")
    plt.ylabel("Prediction")

    for i,entry in enumerate(input_list):
        if entry == "RR":
            shape = "^"
            color_shape = "olivedrab"
            marker_fillstyle = "none"
            markersize=12
        if entry == "SP":
            shape = "x"
            color_shape = "olivedrab"
            marker_fillstyle = "full"
            markersize=12
        if entry == "RR|SP":
            shape = "s"
            color_shape = "brown"
            marker_fillstyle = "full"
            markersize=12
        if entry == "no prediction":
            shape = "|"
            color_shape = "red"
            marker_fillstyle = "full"
            markersize=15
        plt.plot(x[i],y[i],color=color_shape,marker=shape, markersize=markersize,fillstyle=marker_fillstyle)

    y_tick_order = ["RR","RR|SP","SP","no\nprediction"]
    plt.yticks(range(len(y_tick_order)), y_tick_order)

    x_ticks_age = [round(entry-age_at_visit_order[0],2) for entry in age_at_visit_order]
    plt.xticks(list(range(len(input_list))),x_ticks_age,rotation=45)

    for i,(xi,yi) in enumerate(zip(x,y)):
        plt.text(xi+(xi*0.01), yi+(yi*0.01),np.round(max(test_pvalues[i]),2),rotation = 45,size=16)

    fig.tight_layout()

    return fig


def pval_plot_for_slider(json_data):
    return gr.Plot.update(pval_plot(json_data),visible=True)

def pval_plot(json_data):
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False

    test_pvalues = np.array(json_data["test_pval"])
    age_at_visit_order = json_data["age_at_visit_order"]
    test_conformal_output = json_data["test_class"]

    x_label = np.array(age_at_visit_order) - min(age_at_visit_order)
    fig = plt.figure()
    fig.set_size_inches(15, 9)


    rr_pval = test_pvalues[:,:1].squeeze()
    sp_pval = test_pvalues[:,1:].squeeze()
    x_label_redone = copy.deepcopy(x_label)
    rr_pval_orig = copy.deepcopy(rr_pval)
    sp_pval_orig = copy.deepcopy(sp_pval)


    x_label_years = np.array(age_at_visit_order) - min(age_at_visit_order)
    x_label_years = [round(entry,2) for entry in x_label_years]

    plot = plt.plot(rr_pval,color="teal",label="RRMS p-value")
    add_each_point_property_pval_plot(plot,test_pvalues,test_conformal_output)

    plot = plt.plot(sp_pval,color="orange",label="SPMS p-value")
    add_each_point_property_pval_plot(plot,test_pvalues,test_conformal_output)

    plt.xticks(list(range(len(x_label))),x_label_years,rotation=45)
    plt.xlabel("Years since disease onset")
    plt.ylabel("p-value")

    legend_elements = [Line2D([0], [0], marker='_', color='orange', label='SPMS',markerfacecolor='olivedrab', markersize=9),
                        Line2D([0], [0], marker='_', color='teal', label='RRMS',markerfacecolor='tomato', markersize=9)]
    legend1 = plt.legend(handles=legend_elements, loc='upper left',fontsize=16,framealpha=0.0)
    #plt.artist(legend1)

    line_legend = [Line2D([0], [0], marker='^', color='black', label='RR',markerfacecolor='black', markersize=9,fillstyle='none',alpha=0.3),
                      Line2D([0], [0], marker='x', color='black', label='SP',markerfacecolor='black', markersize=9,alpha=0.3),
                      Line2D([0], [0], marker='s', color='black', label='RR | SP',markerfacecolor='black', markersize=9,alpha=0.3),
                      Line2D([0], [0], marker='|', color='black', label='{}',markerfacecolor='black', markersize=15,alpha=0.3)]
    #legend2 = plt.legend(handles=line_legend, loc='upper right',fontsize=16,framealpha=0.0)
    plt.gca().add_artist(legend1)
    #plt.add_artist(legend2)

    fig.tight_layout()
    return fig


def add_each_point_property_pval_plot(plot,pvals,test_conformal_output):
    #print (pvals)
    x,y = plot[0]._x,plot[0]._y
    no_pred_label,rr_label,sp_label,multiton_label = True,True,True,True
    for i,cpi in enumerate(test_conformal_output):
        labelling = False
        x_,y_ = x[i],y[i]
        color_shape = "olivedrab"
        if len(cpi) == 0:
            shape = '|'
            if no_pred_label:
                label= "{}"
                marker_fillstyle = "full"
                no_pred_label,labelling = False,True
                markersize=15
            #color_shape = "red"
        if len(cpi) == 1 and cpi[0] == 0:
            shape = "^"
            if rr_label:
                label= "RR"
                marker_fillstyle = "none"
                rr_label,labelling = False,True
                markersize=12
            #color_shape = "green"
        if len(cpi) == 1 and cpi[0] == 1:
            shape = 'x'
            if sp_label:
                label= "SP"
                marker_fillstyle = "full"
                sp_label,labelling = False,True
                markersize=12
            #color_shape = "green"
        if len(cpi) == 2:
            shape = 's'
            if multiton_label:
                label= "RR | SP"
                marker_fillstyle = "full"
                multiton_label,labelling = False,True
                markersize=12
            #color_shape = "brown"
        if labelling:
            plt.plot(x_,y_,shape,color=color_shape,label=label,fillstyle=marker_fillstyle,markersize=markersize)
        else:
            plt.plot(x_,y_,shape,color=color_shape,fillstyle=marker_fillstyle,markersize=markersize)


def slider_function(slider_value,json_data):
    slider_value_significane = 1- (slider_value/100)
    x_test = np.array(json_data["x_test"])
    age_at_visit_order = json_data["age_at_visit_order"]
    colnames = json_data["colnames"]


    icp = joblib.load(icp_model_path)
    test_pval = icp.predict(x_test)
    test_class = pred_class_truth_table_to_bool(icp.predict(x_test,significance=slider_value_significane))

    json_data = json.loads(json.dumps({"test_class":test_class,
                            "test_pval":test_pval.tolist(),
                            "age_at_visit_order":age_at_visit_order,
                            "x_test":x_test.tolist(),
                            "colnames":colnames}))
    
    
    return p0p1plot_for_slider(json_data),disease_course_plot_for_slider(json_data),pval_plot_for_slider(json_data)

##### PLOTS - END

#####
def pred_class_truth_table_to_bool(pred_class):
    out_list = []
    for entry1 in pred_class:
        current_pred = []
        for i,entry2 in enumerate(entry1):
            if entry2 == True and i == 1:
                current_pred.append(1)
            if entry2 == True and i == 0:
                current_pred.append(0)
        out_list.append(current_pred)
    return out_list

def model_predict(patient_df,slider_value,json_data):
    features =['edss_score','age_at_visit','mono_on_sum','revised_debut_age',
            'monofocal_sum','multifocal_sum','afferent_non_on_sum',
           'steroid_treatment_sum','is_last_relapse_steroid_treated',
            'is_last_relapse_completely_remitted','age_at_relapse', 'age_at_debut_relapse']

    age_at_visit_order = patient_df["age_at_visit"].to_list()
    x_test_df = patient_df[features]
    colnames = list(x_test_df.columns)
    x_test = x_test_df.values
    
    icp = joblib.load(icp_model_path)
    test_pval = icp.predict(x_test)
    
    slider_value_significane = 1- (slider_value/100)
    test_class = pred_class_truth_table_to_bool(icp.predict(x_test,significance=slider_value_significane))

    json_data = json.loads(json.dumps({"test_class":test_class,
                            "test_pval":test_pval.tolist(),
                            "age_at_visit_order":age_at_visit_order,
                            "x_test":x_test.tolist(),
                            "colnames":colnames}))
    
    # All the figures
    p0p1_plot1 = p0p1plot(json_data)

    disease_course_plot_plot2 = disease_course_plot(json_data)

    pval_plot3 = pval_plot(json_data)

    return [gr.Slider.update(visible=True),gr.Plot.update(p0p1_plot1,visible=True),gr.Plot.update(disease_course_plot_plot2,visible=True),gr.Plot.update(pval_plot3,visible=True),json_data]




def segregate_direct_data(input_data):
    visit_data,relapse_data = [],[]
    break_count = 0
    current_row_data = []
    i = 1
    for entry in input_data:
        if entry == "break":
            break_count += 1
            i = 1
            continue

        if break_count == 0:
            if i%2 == 0:
                current_row_data.append(entry)
                visit_data.append(current_row_data)
                current_row_data = []
            if i%2==1:
                current_row_data.append(entry)

        if break_count == 1:
            if i%6 == 0:
                current_row_data.append(entry)
                relapse_data.append(current_row_data)
                current_row_data = []
            else:
                current_row_data.append(entry)
        i += 1

    visit_df = pd.DataFrame(visit_data,columns=["visit_date","edss_score"])
    relapse_df = pd.DataFrame(relapse_data,columns=["relapse_date","monofocal","steroid_treatment",
                                                    "is_last_relapse_completely_remitted","mono_on","afferent_non_on"])

    return visit_df,relapse_df


def segregate_file_data(input_files):
    for files in input_files:
        if "visit.csv" in files.name:
            visit_df = pd.read_csv(files.name)
        if "relapse.csv" in files.name:
            relapse_df = pd.read_csv(files.name)
    return visit_df,relapse_df


def get_patient_df(visit_df,relapse_df):
    # Get patient object by parsing all the dataframes accordingly

    # Make data into necessary format
    visit_df = visit_df.sort_values("visit_date")
    relapse_df = relapse_df.sort_values("relapse_date")


    # maintaining column order
    relapse_df = relapse_df[["relapse_date","debut_relapse_date","monofocal","steroid_treatment"
                             ,"is_last_relapse_completely_remitted","mono_on","afferent_non_on"]]
    relapse_df["monofocal"] = pd.to_numeric(relapse_df["monofocal"])
    relapse_df["steroid_treatment"] = pd.to_numeric(relapse_df["steroid_treatment"])
    relapse_df["is_last_relapse_completely_remitted"] = pd.to_numeric(relapse_df["is_last_relapse_completely_remitted"])
    relapse_df["mono_on"] = pd.to_numeric(relapse_df["mono_on"])
    relapse_df["afferent_non_on"] = pd.to_numeric(relapse_df["afferent_non_on"])

     # Convert data to single df
    visit_df["age_at_visit"] =  np.round((visit_df["visit_date"]-visit_df["dob_year"]) / np.timedelta64(1, 'Y'),3)
    visit_df["revised_debut_age"] =  np.round((visit_df["debut_date"]-visit_df["dob_year"]) / np.timedelta64(1, 'Y'),3)
    for index,row in visit_df.iterrows():
        dob_year = row["dob_year"]
        visit_date = row["visit_date"]

        # For SKOV
        sub_relapse_df = relapse_df[relapse_df["relapse_date"] <= visit_date]
        if len(sub_relapse_df) > 0:
            sub_relapse_values = sub_relapse_df.values
            relapse_date,debut_relapse_date = sub_relapse_values[-1][0],sub_relapse_values[-1][1]
            age_at_relapse = np.round((relapse_date-dob_year) / np.timedelta64(1, 'Y'),3)
            age_at_debut_relapse = np.round((debut_relapse_date-dob_year) / np.timedelta64(1, 'Y'),3)
            monofocal_sum = sum(sub_relapse_values[:,2] >= 0)
            multifocal_sum = len(sub_relapse_values[:,2] >= 0) - monofocal_sum
            
            if monofocal_sum == 0 and  np.all(sub_relapse_values[:,2]==-1):
                monofocal_sum = -1
                multifocal_sum = -1

            steroid_treatment_sum = sum(sub_relapse_values[:,3] >= 0)
            if steroid_treatment_sum == 0 and  np.all(sub_relapse_values[:,3]==-1):
                steroid_treatment_sum = -1
                
            is_last_relapse_steroid_treated = sub_relapse_values[:,3][-1]
                
            is_last_relapse_completely_remitted = sub_relapse_values[:,4][-1]
            
            mono_on_sum = sum(sub_relapse_values[:,5] >= 0)
            if mono_on_sum == 0 and  np.all(sub_relapse_values[:,5]==-1):
                mono_on_sum = -1
                
            afferent_non_on_sum = sum(sub_relapse_values[:,6] >= 0)
            if afferent_non_on_sum == 0 and  np.all(sub_relapse_values[:,6]==-1):
                afferent_non_on_sum = -1
        else:
            age_at_debut_relapse,monofocal_sum,multifocal_sum,steroid_treatment_sum,age_at_relapse,is_last_relapse_steroid_treated,is_last_relapse_completely_remitted,mono_on_sum,afferent_non_on_sum = -1,-1,-1,-1,-1,-1,-1,-1,-1

        visit_df.loc[index,"age_at_debut_relapse"] = age_at_debut_relapse
        visit_df.loc[index,"monofocal_sum"] = monofocal_sum
        visit_df.loc[index,"multifocal_sum"] = multifocal_sum
        visit_df.loc[index,"steroid_treatment_sum"] = steroid_treatment_sum
        visit_df.loc[index,"age_at_relapse"] = age_at_relapse
        visit_df.loc[index,"is_last_relapse_steroid_treated"] = is_last_relapse_steroid_treated
        visit_df.loc[index,"is_last_relapse_completely_remitted"] = is_last_relapse_completely_remitted
        visit_df.loc[index,"mono_on_sum"] = mono_on_sum
        visit_df.loc[index,"afferent_non_on_sum"] = afferent_non_on_sum

    visit_df["patient_code"] = 1
    visit_df["progress_during_visit"] = "RR"

    return visit_df


def round_edss(input_num): # To round of interger with a step of 0.5
    if abs((input_num%1)-0.5) <= 0.25:
        return ((input_num//1) + 0.5)
    if 5 > abs((input_num%1)-0.5) > 0.25:
        if (input_num%1)-0.5 > 0:
            return ((input_num//1)+1)
        else:
            return ((input_num//1))
    return input_num


def pandas_date_string_parsing(input_series):
    return input_series.str.replace("-","/")

def parse_data(dob_value,debut_date,debut_relapse_date,visit_df,relapse_df):

    # Removing non-necessay entries which are -1
    visit_df = visit_df[(visit_df["visit_date"] != "")]
    visit_df = visit_df[(visit_df["visit_date"] != "-1")]
    relapse_df = relapse_df[(relapse_df["relapse_date"] != "")]
    relapse_df = relapse_df[(relapse_df["relapse_date"] != "-1")]

    # Clip values between boundries
    visit_df["edss_score"] = visit_df["edss_score"].clip(0,10)
    visit_df["edss_score"] = visit_df["edss_score"].apply(lambda x:round_edss(x)) # Rounding edss to nearest int or .5

    visit_df["dob_year"] = dob_value
    visit_df["debut_date"] = debut_date
    relapse_df["debut_relapse_date"] = debut_relapse_date

    # To replace - in date with /
    visit_df["visit_date"] = pandas_date_string_parsing(visit_df["visit_date"])
    visit_df["dob_year"] = pandas_date_string_parsing(visit_df["dob_year"])
    visit_df["debut_date"] = pandas_date_string_parsing(visit_df["debut_date"])
    relapse_df["debut_relapse_date"] = pandas_date_string_parsing(relapse_df["debut_relapse_date"])
    relapse_df["relapse_date"] = pandas_date_string_parsing(relapse_df["relapse_date"])
    ##

    visit_df["visit_date"] =  pd.to_datetime(visit_df['visit_date'], format='%m/%Y')
    visit_df["dob_year"] =  pd.to_datetime(visit_df['dob_year'], format='%Y')
    visit_df["debut_date"] =  pd.to_datetime(visit_df['debut_date'], format='%m/%Y')
    
    relapse_df["debut_relapse_date"] =  pd.to_datetime(relapse_df['debut_relapse_date'], format='%m/%Y')
    relapse_df["relapse_date"] =  pd.to_datetime(relapse_df['relapse_date'], format='%m/%Y')
    return get_patient_df(visit_df,relapse_df)


def direct_predict(dob_value,debut_date,debut_relapse_date,slider_value,json_data,*input_values):
    visit_df,relapse_df = segregate_direct_data(input_values)
    patient_df = parse_data(dob_value,debut_date,debut_relapse_date,visit_df,relapse_df)
    
    return model_predict(patient_df,slider_value,json_data)


def file_predict(dob_value,debut_date,debut_relapse_date,slider_value,json_data,uploaded_files):
    visit_df,relapse_df = segregate_file_data(uploaded_files)
    patient_df = parse_data(dob_value,debut_date,debut_relapse_date,visit_df,relapse_df)

    return model_predict(patient_df,slider_value,json_data)
