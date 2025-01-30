# OFIQ-Python

OFIQ model checkpoints and test images can be downloaded from <https://standards.iso.org/iso-iec/29794/-5/ed-1/en/>.

## Setup

We recommend [miniconda](https://docs.anaconda.com/miniconda/) to set up your conda environment:

```bash
conda env create -n $YOUR_ENV_NAME -f environment.yml
conda activate $YOUR_ENV_NAME
pip install -r requirements.txt
pre-commit install
```

## Run Jupyter notebook

You can run the Jupyter notebooks in the `notebooks/` directory locally using either [Jupyter](https://jupyter.org/) or in the integrated Jupyter environment in [Visual Studio Code](https://code.visualstudio.com/).

Additionally, we provide a notebook with `-colab` in the filename that you can run in the Browser by clicking on the `Open in Colab` button inside the notebook.
