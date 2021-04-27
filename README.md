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
## Overview

This tutorial is designed to demonstrate a typical machine learning development workflow using trainML GPUs.  It walks through how to create a [trainML Dataset](https://app.trainml.ai/datasets), build an initial model using a [trainML Notebook](https://app.trainml.ai/jobs/notebook), run parallel hyperparameter tuning experiments using [trainML Training Jobs](https://app.trainml.ai/jobs/training), save the results of a marathon training job to a resuable [trainML Model](https://app.trainml.ai/models), use that model to run a [trainML Inference Job](https://app.trainml.ai/jobs/inference) on a batch of new images, and receive the results on your local computer.  The data used by this tutorial consists of DICOM files of chest radiographs and their associated labels from the Kaggle [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).  The model code was largely adapted from [Kaggle notebooks](https://www.kaggle.com/giuliasavorgnan/0-123-lb-pytorch-unet-run-on-google-cloud) by Guilia Savorgnan.  The following changes were made to the original code to better facilitate the tutorial

- Changed all directory path references to match the location of the data, temp, and output directories in the trainML job environment.
- Converted the notebooks into python scripts with an [argparse](https://docs.python.org/3/howto/argparse.html) command line interface.
- Exposed and implemented additional hyperparameter settings.
- Added rudimentary tensorboard logging.
- Changed to save the predictions as JSON and save annotated images as PNGs.

> This code is for pedagogical purposes only.  It is not meant as an example of a high performing or efficient model.  It's only purpose is to show the various ways the trainML capabilities can be utilized in the model development process.  Do NOT use this in production.

![Example Prediction](example.png)

### Prerequisites

Before beginning this tutorial, ensure that you have satisfied the following prerequisites.

- A valid [trainML account](https://auth.trainml.ai/login?response_type=code&client_id=536hafr05s8qj3ihgf707on4aq&redirect_uri=https://app.trainml.ai/auth/callback) with a non-zero [credit balance](https://docs.trainml.ai/reference/billing-credits/)
- Local connection capability [prerequisites](https://docs.trainml.ai/reference/third-party-keys/#kaggle-keys)
- Valid [Kaggle](https://www.kaggle.com) account with [Kaggle Keys](https://docs.trainml.ai/reference/third-party-keys/#kaggle-keys) configured in your trainML account.
- Accepted the competition terms of the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) on Kaggle.

## Model Development


## Model Training


## Inference Pipeline


## Additional Information

