<h1 align="center">
<a href="https://msp-tracker.serve.scilifelab.se/"> 
MSP-Tracker
</a></h1>
<h2 align="center">(Multiple Sclerosis Progression-tracker)</h2>

---
### Setting up the environment
To create and activate the environment. <br>
```bash
conda env create -f environment.yml
conda activate websmsreg
```
To export the conda environment to Jupyter Notebook. <br>
```bash
python -m ipykernel install --user --name=websmsreg
```
<br>

---


### Data preprocessing and creating data splits
The data can be collected from Swedish MS REGistry (SMSREG). The data shall be kept in a designated folder. The path to the data can be provided in the [notebook (data_cleaning_and_splitting.ipynb)](scripts/data_cleaning_and_splitting.ipynb) under the input section.   <br>
### Training
Code for training and evaluation of both the model and conformal prediction is given in [notebook (random_forest_cp.ipynb)](scripts/random_forest_cp.ipynb) <br>
### [MSP-Tracker - Take me to the website](https://msp-tracker.serve.scilifelab.se/) <br>
The model and scripts used for the website model are available in the [folder gradio](gradio).<br>
To run the website locally, activate the conda environment and run

```bash
python3 gradio/app.py
```

## Citation
Please cite:<br>
>Sreenivasan, A.P., Vaivade, A., Noui, Y. et al. Conformal prediction enables disease course prediction and <br>
>allows individualized diagnostic uncertainty in multiple sclerosis. npj Digit. Med. 8, 224 (2025). <br>
>https://doi.org/10.1038/s41746-025-01616-z
