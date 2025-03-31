import sys
import gradio as gr

from . import app_utils as apu

max_textboxes = 25
def main(dummy_checkbox):

    if dummy_checkbox: # If enable dummy data is true

        visit_output = []
        visit_output.append(gr.Textbox.update(visible=True,value="03/1991"))
        visit_output.append(gr.Number.update(visible=True,value=1))

        visit_output.append(gr.Textbox.update(visible=True,value="11/1992"))
        visit_output.append(gr.Number.update(visible=True,value=2))

        visit_output.append(gr.Textbox.update(visible=True,value="06/1994"))
        visit_output.append(gr.Number.update(visible=True,value=3.5))

        visit_output.append(gr.Textbox.update(visible=True,value="12/1996"))
        visit_output.append(gr.Number.update(visible=True,value=4))

        visit_output.append(gr.Textbox.update(visible=True,value="10/2000"))
        visit_output.append(gr.Number.update(visible=True,value=4.5))

        visit_output.append(gr.Textbox.update(visible=True,value="09/2002"))
        visit_output.append(gr.Number.update(visible=True,value=4.5))

        visit_output.append(gr.Textbox.update(visible=True,value="11/2003"))
        visit_output.append(gr.Number.update(visible=True,value=5))

        visit_output.append(gr.Textbox.update(visible=True,value="12/2007"))
        visit_output.append(gr.Number.update(visible=True,value=6))

        visit_output.append(gr.Textbox.update(visible=True,value="05/2011"))
        visit_output.append(gr.Number.update(visible=True,value=7.5))
        for i in range(max_textboxes):

            if i >= 9:
                visit_output.append(gr.Textbox.update(visible=False,value=""))
                visit_output.append(gr.Number.update(visible=False,value=None))



        relapse_output = []
        relapse_output.append(gr.Textbox.update(visible=True,value="04/1990"))
        relapse_output.append(gr.Dropdown.update(visible=True,value=1))
        relapse_output.append(gr.Dropdown.update(visible=True,value=0))
        relapse_output.append(gr.Dropdown.update(visible=True,value=1))
        relapse_output.append(gr.Dropdown.update(visible=True,value=0))
        relapse_output.append(gr.Dropdown.update(visible=True,value=1))

        relapse_output.append(gr.Textbox.update(visible=True,value="08/1996"))
        relapse_output.append(gr.Dropdown.update(visible=True,value=0))
        relapse_output.append(gr.Dropdown.update(visible=True,value=1))
        relapse_output.append(gr.Dropdown.update(visible=True,value=1))
        relapse_output.append(gr.Dropdown.update(visible=True,value=1))
        relapse_output.append(gr.Dropdown.update(visible=True,value=0))

        relapse_output.append(gr.Textbox.update(visible=True,value="03/1999"))
        relapse_output.append(gr.Dropdown.update(visible=True,value=1))
        relapse_output.append(gr.Dropdown.update(visible=True,value=-1))
        relapse_output.append(gr.Dropdown.update(visible=True,value=1))
        relapse_output.append(gr.Dropdown.update(visible=True,value=0))
        relapse_output.append(gr.Dropdown.update(visible=True,value=0))
        for i in range(max_textboxes):
            if i >= 3:
                relapse_output.append(gr.Textbox.update(visible=False,value=""))
                relapse_output.append(gr.Dropdown.update(visible=False,value=0))
                relapse_output.append(gr.Dropdown.update(visible=False,value=0))
                relapse_output.append(gr.Dropdown.update(visible=False,value=0))
                relapse_output.append(gr.Dropdown.update(visible=False,value=0))
                relapse_output.append(gr.Dropdown.update(visible=False,value=0))

        return  [gr.Textbox.update(visible=True,value=1950), gr.Textbox.update(visible=True,value="09/1989"), gr.Textbox.update(visible=True,value="04/1990")] +  [gr.Number.update(value=9)] + visit_output + [gr.Number.update(value=3)] + relapse_output



    else: # If disable dummy data
        visit_output = []
        for i in range(max_textboxes):
            if i < 1:
                visit_output.append(gr.Textbox.update(visible=True,value=""))
                visit_output.append(gr.Number.update(visible=True,value=None))
            else:
                visit_output.append(gr.Textbox.update(visible=False,value=""))
                visit_output.append(gr.Number.update(visible=False,value=None))

        relapse_output = []
        for i in range(max_textboxes):
            if i < 1:
                relapse_output.append(gr.Textbox.update(visible=True,value=""))
                relapse_output.append(gr.Dropdown.update(visible=True,value="0"))
                relapse_output.append(gr.Dropdown.update(visible=True,value="0"))
                relapse_output.append(gr.Dropdown.update(visible=True,value="0"))
                relapse_output.append(gr.Dropdown.update(visible=True,value="0"))
                relapse_output.append(gr.Dropdown.update(visible=True,value="0"))
            else:
                relapse_output.append(gr.Textbox.update(visible=False,value=""))
                relapse_output.append(gr.Dropdown.update(visible=False,value="0"))
                relapse_output.append(gr.Dropdown.update(visible=False,value="0"))
                relapse_output.append(gr.Dropdown.update(visible=False,value="0"))
                relapse_output.append(gr.Dropdown.update(visible=False,value="0"))
                relapse_output.append(gr.Dropdown.update(visible=False,value="0"))

        return  [gr.Textbox.update(visible=True,value=""), gr.Textbox.update(visible=True,value=""), gr.Textbox.update(visible=True,value="")] +[gr.Number.update(value=1)] + visit_output + [gr.Number.update(value=1)] + relapse_output



def dummy_files(checkbox_value):
    base_path = "assets/files/"
    if checkbox_value:
        return [gr.Textbox.update(visible=True,value=1950), gr.Textbox.update(visible=True,value="09/1989"), gr.Textbox.update(visible=True,value="04/1990"),gr.Files.update(value=[base_path + "visit.csv",base_path + "relapse.csv"])]
    else:
        return  [gr.Textbox.update(visible=True,value=""), gr.Textbox.update(visible=True,value=""), gr.Textbox.update(visible=True,value=""),gr.Files.update(value=None)]


def help_checkbox(checkbox_value):
    if checkbox_value:
        return (gr.Image.update(visible=True))
    else:
        return (gr.Image.update(visible=False))




def citation_checkbox(checkbox_value):
    if checkbox_value:
        return (gr.Textbox.update(visible=True))
    else:
        return (gr.Textbox.update(visible=False))
