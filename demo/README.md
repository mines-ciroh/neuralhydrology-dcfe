# Inspecting Streamflow Predictions, Internal States, and Parameters for Hybrid Models in Neural Hydrology

## Running the Application

To run the user interface, ensure you have all of the dependencies installed in `environments/requirements.txt` via a package manager of your choosing (we recommend `conda` or `uv`).

Next, simply run `python3 demo/app.py` in your terminal, then follow one of the URLs to the site.

## Overview

The two hybrid models we have implemented in NeuralHydrology are `SHM` and `dCFE`, which stand for:

- Simple Hydrology Model
- Differentiable Conceptual Functional Equivalent

The models are denoted as "hybrid". This is because they are composed of (1) a machine learning model and (2) a process-based or physical hydrological model. The machine learning model is a LSTM neural network in our case.

The LSTM predicts the input parameters for the physical hydrology model for the entire (train/validation/test) period. Then, those input parameters are passed through the physical model's solver. Once streamflow is computed for the period, the loss is computed from observations, which is then used to update LSTM weights.

Before predicting the streamflow, the hybrid models are "spun-up" for a period of time to allow their inputs and states to reach physically meaningful values. We call this period the "spin-up" period.

In addition, we offer three modes that change how the LSTM input parameters are fed into the physical model:

- Dynamic Mode
- Oracle Averaging Mode
- Operational Averaging Mode

**Dynamic Mode** is simply the raw LSTM output parameters. This is the baseline.

**Oracle Averaging Mode** is named as such because it "looks into the future", meaning that it takes the mean of the LSTM output parameters for both the spin-up and prediction periods.

**Operational Averaging Mode** takes the mean of the LSTM output parameters for only the spin-up period.
