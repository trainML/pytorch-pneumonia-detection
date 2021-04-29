<!--- Licensed to the trainML under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The trainML this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<div align="center">
  <a href="https://www.trainml.ai/"><img src="https://www.trainml.ai/static/img/trainML-logo-purple.png"></a><br>
</div>


trainML Tutorials - PyTorch Pneumonia Detection Pipeline
=====

![Example Prediction](example.png)

## Overview

This tutorial is designed to demonstrate a typical machine learning development workflow using trainML GPUs.  It walks through how to create a [trainML Dataset](https://app.trainml.ai/datasets), build an initial model using a [trainML Notebook](https://app.trainml.ai/jobs/notebook), run parallel hyperparameter tuning experiments using [trainML Training Jobs](https://app.trainml.ai/jobs/training), save the results of a marathon training job to a resuable [trainML Model](https://app.trainml.ai/models), use that model to run a [trainML Inference Job](https://app.trainml.ai/jobs/inference) on a batch of new images, and receive the results on your local computer.  The data used by this tutorial consists of DICOM files of chest radiographs and their associated labels from the Kaggle [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).  The model code was largely adapted from [Kaggle notebooks](https://www.kaggle.com/giuliasavorgnan/0-123-lb-pytorch-unet-run-on-google-cloud) by Guilia Savorgnan.  The following changes were made to the original code to better facilitate the tutorial

- Changed all directory path references to match the location of the data, temp, and output directories in the trainML job environment.
- Converted the notebooks into python scripts with an [argparse](https://docs.python.org/3/howto/argparse.html) command line interface.
- Exposed and implemented additional hyperparameter settings.
- Added rudimentary tensorboard logging.
- Changed to save the predictions as JSON and save annotated images as PNGs.

> This code is for pedagogical purposes only.  It is not meant as an example of a high performing or efficient model.  It's only purpose is to show the various ways the trainML capabilities can be utilized in the model development process.  Do NOT use this in production.

### Prerequisites

Before beginning this tutorial, ensure that you have satisfied the following prerequisites.

- A valid [trainML account](https://auth.trainml.ai/login?response_type=code&client_id=536hafr05s8qj3ihgf707on4aq&redirect_uri=https://app.trainml.ai/auth/callback) with a non-zero [credit balance](https://docs.trainml.ai/reference/billing-credits/)
- Local connection capability [prerequisites](https://docs.trainml.ai/reference/third-party-keys/#kaggle-keys)
- Valid [Kaggle](https://www.kaggle.com) account with [Kaggle Keys](https://docs.trainml.ai/reference/third-party-keys/#kaggle-keys) configured in your trainML account.
- Accepted the competition terms of the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) on Kaggle.

## Model Development

### Dataset Staging

The first step in building an initial model is to load the training data as a [trainML Dataset](https://docs.trainml.ai//reference/datasets).  The training data can be viewed [here](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data).  In order to create a trainML dataset from Kaggle competition data, login to the trainML web interface and navigate to the [Datasets Dashboard](https://app.trainml.ai/datasets).  Click the `Create` button from the dashboard.  Enter a memorable name in the name field (e.g. `RSNA Pneumonia Detection`), select `Kaggle` as the source type, select `Competition` as the type, and enter `rsna-pneumonia-detection-challenge` as the path.  Click the `Create` button to begin the dataset creation.  Once the dataset changes to the `ready` status, it can be used in the subsequent steps.

> If the `Kaggle` option is disabled, you have not yet configured your Kaggle API keys in your trainML account.  Follow the instructions [here](https://docs.trainml.ai/reference/third-party-keys/#kaggle-keys) to proceed.

The RSNA Pneumonia Detection dataset has two sets of images `stage_2_train_images` and `stage_2_test_images`.  Only the "train" images have labels.  We will use the "train" images during the training process and the "test" images to demonstrate the inference process.  For the inference process, you should also download the Kaggle data to your local computer using the command: 

`kaggle competitions download -c rsna-pneumonia-detection-challenge`

Once the download completes, unzip the file and save the contents of the `stage_2_test_images` folder to a memorable location (the tutorial uses `~/rsna-pneumonia-detection-challenge/new_images`).  The rest of the data can be deleted.

### Initial Model Creation

The easiest way to start a new project is with a [trainML Notebook](https://docs.trainml.ai/getting-started/running-notebook/).  Navigate to the [Notebooks Dashboard](https://app.trainml.ai/notebook) and click the `Create` button.  Input a memorable name as the job name and select an available GPU Type (the code in this tutorial assumes a `RTX 2080 Ti`).  Expand the `Data` section and click `Add Dataset`.  Select `My Dataset` as the dataset type and select the dataset you created in the previous section from the list (e.g. `RSNA Pneumonia Detection`).  Expand the `Model` section and specify the project's git repository url `https://github.com/trainML/pytorch-pneumonia-detection.git` to automatically download the tutorial's model code.  Click `Next` to view a summary of the new notebook and click `Create` to start the notebook.

Once the notebook reaches the `running` state, click the `Open` to access the notebook instance.  Inside the [Jupyter Lab](https://jupyter.org) environment, the file browser pane on the left will show two directories, `input` and `models`.  The `input` folder contains the RSNA Pneumonia dataset and the `models` folder contains the code from the git repository.  Double click on the models folder and open the `eda-adapted` notebook.  This notebook contains some exploratory data analysis on the dataset.  The original source is located [here](https://www.kaggle.com/giuliasavorgnan/start-here-beginner-intro-to-lung-opacity-s1), it was only modified to direct file path variables to the correct location with the trainML job environment.  It also generates a features file that is required for the model training notebook.  Either run this notebook to generate the file or run `python data_processing.py` from a terminal tab.

Once the `train.csv` appears in the file explorer in the `models` folder, open the `pytorch-pneumonia-detection` notebook.  This notebook contains the model training and evaluation code.  You can find the original [here](https://www.kaggle.com/giuliasavorgnan/0-123-lb-pytorch-unet-run-on-google-cloud).  You can either review the stored results or run all cells to observe the training yourself.  **To shorten the duration, change the `Debug` variable to True**

> Take note of how the [trainML environment variables](https://docs.trainml.ai/reference/environment-variables/) are used to define the different datapaths in both the notebooks and as the default arguments in the data_processing.py script.  This is the recommended way to define file locations in models when using the trainML job environment.

Continue to explore the notebook design and job environment as desired.  In most real projects, the objective of the notebook stage is to ensure the data is being loaded correctly and the model code is executing correctly before moving on to longer duration training experiments.

## Model Training

### Adapting the Notebook for Training


### Parallel Hyperparamter Search


### Marathon Training


## Inference Pipeline


## Additional Information

