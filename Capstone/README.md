# Content: Udacity Capstone
## Project: [Porto Seguro’s Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) in Kaggle Competition

### Install

This project requires **Python 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [xgboost](https://xgboost.readthedocs.io/)
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

### Code

**_For Jupyter version_** 
the manin code is provided in the `Potro.ipynb` notebook file. You will also be required to use the included `helper.py` Python file, the `train.csv` and `test.csv` dataset file,which you have to download from [Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data) into the input folder, to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the. During the operation of `Potro.ipynb` in , the defualt output will be created in model folder. Note that the code included in `helper.py` is meant to be used out-of-the-box and not intended for developer to manipulate. If you are interested in `helper.py`, please feel free to explore this Python file. 

**_For Google Colud_**
 the manin code is provided in the `trainer/task.py` python file. You may need [Google Command Line Tool](https://cloud.google.com/sdk/) to submmit job to Google Cloud ML Engine. `excercute.sh` content the command line for you to submmit to google Cloud. You will also be required to use the included `helper.py` Python file, the `train.csv` and `test.csv` dataset file,which you have to download from [Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data) into the input folder, to complete your work. `setup.py` provide the requirement of package that you need for runing Cloud ML Engine. Change `config/config.yml` to access the resource from Google Cloud like python version and service GPU request.


### Run

In a terminal or command window, navigate to the top-level project directory `Jupyter_Version/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Potro.ipynb
```  
or
```bash
jupyter notebook Potro.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

## Data

You can download data from [Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)
In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.


