import sys
sys.path.append("assets")
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gradio as gr

### CUSTOM PACKAGES
import scripts.app_utils as apu
import scripts.enable_dummy_data as dummy
import scripts.shaputils as shapu

max_textboxes = 25

def lambda_fuction(x):
    return x[1]


with gr.Blocks() as demo:

    gr.Image(value="assets/imgs/header.png",show_label=True)


    citation_box = gr.Checkbox(value=True,visible=True,interactive=True,label="How to cite?")
    citation_text = gr.Textbox("Conformal prediction enables disease course prediction and allows individualised diagnostic uncertainty in multiple sclerosis \n\
Akshai Parakkal Sreenivasan, Aina Vaivade, Yassine Noui, Payam Emami Khoonsari, Joachim Burman, Ola Spjuth, Kim Kultima*\n\
Status: Submitted")
    citation_box.change(dummy.citation_checkbox,[citation_box],[citation_text])


    manual_file = gr.File(value="assets/files/manual.pdf",label="Manual for the website",interactive=False)

    # Set static rows
    visibility = True
    with gr.Row():
        dob_value = gr.Textbox(label="Birth year (YYYY)",interactive =True,visible=True,value="")
        debut_date = gr.Textbox(label="Disease onset date (MM/YYYY)",interactive =True,visible=True,value="",info="Provide diagnosis date if onset date is unknown")
        debut_relapse_date = gr.Textbox(label="Date of first relapse (MM/YYYY)",interactive =True,visible=True,value="")




    with gr.Tab("Direct predict"):
        dummy_data_checkbox = gr.Checkbox(value=False,visible=True,interactive=True,label="Input dummy data")
        # Visit information
        with gr.Row():
            visit_input_length = gr.Number(label="Number of visits",interactive =True,visible=True,value=1)
            visit_size_submit_button = gr.Button("Set")
        # Set variable rows
        visit_values_row = []
        for i in range(max_textboxes):
            if i < int(visit_input_length.value):
                visibility = True
            else:
                visibility = False

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        visit_date = gr.Textbox(label="Visit date (MM/YYYY)",interactive =True,visible=visibility,value="")
                        edss_value = gr.Number(label="EDSS Value (Number between 0 and 10)",precision=1,interactive =True,visible=visibility)
                visit_values_row.append(visit_date)
                visit_values_row.append(edss_value)

        # On button click make necessary changes by updating visibility of the rows
        visit_size_submit_button.click(apu.hide_visit_row,
                                       visit_input_length,
                                       visit_values_row)

        # relapse information
        with gr.Row():
            relapse_input_length = gr.Number(label="Number of relapse entries",interactive =True,visible=True,value=1)
            relapse_size_submit_button = gr.Button("Set")

        # Set variable rows
        relapse_values_row = []
        for i in range(max_textboxes):
            if i < int(relapse_input_length.value):
                visibility = True
            else:
                visibility = False

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        visit_relapse_date = gr.Textbox(label="Relapse date (MM/YYYY)",interactive =True,visible=visibility,value="")
                        monofocal_relapse = gr.Dropdown(["-1","0","1"],label="Mono/multi focal relapse \n(0-monofocal, 1-multifocal, -1-data not available)",interactive =True,visible=visibility,value="0")
                        steroid_treated = gr.Dropdown(["-1","0","1"],label="Steroid treatment \n(0-steroid untreated, 1-steroid treated, -1-data not available)",interactive =True,visible=visibility,value="0")
                        relapse_remission = gr.Dropdown(["-1","0","1"],label="Did the symptoms completely resolve? \n(0-No, 1-Yes, -1-data not available)",interactive =True,visible=visibility,value="0")
                        mono_on = gr.Dropdown(["-1","0","1"],label="Optic neuritis? (mono on) \n(0-No, 1-Yes, -1-data not available)",interactive =True,visible=visibility,value="0")
                        afferent_non_on = gr.Dropdown(["-1","0","1"],label="Other sensory symptoms? (afferent non on) \n(0-No, 1-Yes, -1-data not available)",interactive =True,visible=visibility,value="0")

                relapse_values_row.append(visit_relapse_date)
                relapse_values_row.append(monofocal_relapse)
                relapse_values_row.append(steroid_treated)
                relapse_values_row.append(relapse_remission)
                relapse_values_row.append(mono_on)
                relapse_values_row.append(afferent_non_on)
        # On button click make necessary changes by updating visibility of the rows
        relapse_size_submit_button.click(apu.hide_relapse_row,
                                       relapse_input_length,
                                       relapse_values_row)

        dummy_data_checkbox.change(dummy.main,
                          inputs=[dummy_data_checkbox],
                          outputs= [dob_value, debut_date, debut_relapse_date] +
                                   [visit_input_length] + visit_values_row + \
                             [relapse_input_length] + relapse_values_row)


        predict_button_direct = gr.Button("Predict")


    with gr.Tab("From file"):
        dummy_files_checkbox = gr.Checkbox(value=False,visible=True,interactive=True,label="Input dummy data")
        base_path = "assets/files/"
        template_files = gr.File(value=[base_path + "visit.csv",base_path + "relapse.csv"],file_count="multiple",label="Template files",interactive=False)

        uploaded_files =  gr.Files(file_types=[".csv"],label="Upload Visit, and Relapse files")
        predict_button_file = gr.Button("Predict")
        dummy_files_checkbox.change(dummy.dummy_files,[dummy_files_checkbox],[dob_value, debut_date, debut_relapse_date,uploaded_files])


    with gr.Row():
        with gr.Tab("Predictions"):
            # Output boxes to put the result
            # The initial conformal prediction is with 0.01 significance
            conformal_confidence = 90
            slider = gr.Slider(minimum=0.1,maximum=100,step=0.1,value=conformal_confidence,label="Confidence level",visible=False,interactive=True)

            help_check_box = gr.Checkbox(value=True,visible=True,interactive=True,label="Show plot interpretation")

            plot_explanation = gr.Image(value="assets/imgs/plot_explanation.png",show_label=True,shape=(5,1))
            help_check_box.change(dummy.help_checkbox,[help_check_box],[plot_explanation])


            plot1 = gr.Plot(label="Diseaes trajectory plot",visible=False)
            plot2 = gr.Plot(label="Disease course plot 1",visible=False)
            plot3 = gr.Plot(label="Disease course plot 2",visible=False)

            json_data = gr.JSON(label="Top Scores",visible=False)
            
            dummy_textbox1 = gr.Textbox(value="break",visible=False)

            predict_button_direct.click(apu.direct_predict,
                                        [dob_value,debut_date,debut_relapse_date,slider,json_data] + visit_values_row  + [dummy_textbox1] + relapse_values_row,
                                        [slider,plot1,plot2,plot3,json_data]
                                    )

            predict_button_file.click(apu.file_predict,
                                        [dob_value,debut_date,debut_relapse_date,slider,json_data,uploaded_files],
                                        [slider,plot1,plot2,plot3,json_data]
                                    )


            # Slider to adjust the conformal prediction plot
            slider.change(apu.slider_function, [slider,json_data],[plot1,plot2,plot3])
            
        with gr.Tab("Feature importance"):
            
            shap_message = gr.Textbox(label="**IMPORTANT**", value="*** RUN PREDICTIONS BEFORE CALCULATING FEATURE IMPORTANCE ***",visible=True)
            shap_button = gr.Button("Calculate feature importance")
            
            shap_barplot = gr.Plot(label="Relative feature importance - Barplot",visible=False)
            shap_beeswarmplot = gr.Plot(label="Overall feature importance - Beeswarm plot",visible=False)
            
            shap_forceplot_list = []
            for i in range(max_textboxes):
                shap_forceplot_list.append(gr.Plot(label="Individual feature importance for datapoint " + str(i+1),visible=False))
                
            shap_button.click(shapu.main,
                                        [json_data],
                                        [shap_barplot,shap_beeswarmplot,json_data] + shap_forceplot_list
                                    )

if __name__ == "__main__":
   demo.launch(share=False)
