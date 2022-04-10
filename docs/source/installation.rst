************
Installation
************

Using Fibermagic with Google Colab
==================================

Google Colab allows you to write and execute arbitrary python code using Jupyter Notebooks through your browser.
We recommend using Google Colab to users who are Newbies and want to start learning fiber photometry analysis.
The advantage of Google Colab is that no further installation is required.

**Start a example notebook**

We provide several example notebooks and datasets. Go to our `Google Drive folder <https://drive.google.com/drive/folders/1fxuuiTTitnL-2ydJRoqUXw23Eg3XQPL6?usp=sharing>`_
and open one of the example notebooks provided. You can open a Colab by double-clicking on the .ipynb file.

**Having trouble with opening?** Some users have issues with directly opening a colab notebook from their Google Drive.
If you experience troubles, go to `Google Colab <https://colab.research.google.com/>`_, click on the tab "GitHub" and paste
"https://github.com/Goreg12345/fibermagic/tree/master/fibermagic/Examples" into the url input box. Then, you should find and be able to open the example notebooks.

.. image:: _static/installation/colab.PNG
    :width: 800

**Start a notebook from scratch**

1. Open a new `Google Colab <https://colab.research.google.com/>`_.
2. Paste :code:`!pip install fibermagic` into the first cell and hit Shift+Enter to execute the cell.
3. Load your data from your Google Drive. Paste this code into the second cell and follow the instructions.

.. code-block:: python

    from google.colab import drive
    drive.mount('/gdrive')

4. Find the location of your dataset in Colab and set the project path variable :code:`PROJECT_PATH = r"/gdrive/MyDrive/Fibermagic Tutorial/Data Processing Exercise/fdrd2xadora_PR_NAcc"`

.. image:: _static/installation/folder.PNG
    :width: 250

5. Your are ready to use fibermagic for your analysis!

Local Installation using Anaconda
=================================

Follow these instructions if you like to use fibermagic on your local computer. We recommend this if you have basic python and programming knowledge and want to use fibermagic for various projects.

**Do once in a lifetime:**

1. Download and install `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_.

**Do for every project:**

1. Create a new anaconda environment. An environment is a place where you can install software specific for a project.
This is important because for different projects you may need different packages and different versions. For example, you might need Python 3.7 for one of your projects and Python 3.10 for another.
An environment prevents you from damaging software of your first project by working on your second project.

.. code-block:: bash

    conda create -n my_experiment python
    conda activate my_experiment

Replace "my_experiment" with your name of your experiment

2. Install fibermagic and other important libs like pandas, numpy and plotting libraries.

.. code-block:: bash

    pip install fibermagic

3. [optional] If you like to use jupyter notebooks, install jupyter notebook and add the environment as a python kernel to jupyter.

.. code-block:: bash

    pip install notebook
    conda install ipykernel
    ipython kernel install --user --name="Python (fibermagic)"
    jupyter notebook  # this starts a jupyter notebook

4. You are ready to use fibermagic!
