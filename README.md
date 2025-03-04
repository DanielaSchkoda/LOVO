# Cross-validating causal discovery via Leave-One-Variable-Out
This is the code to reproduce experiments the from the paper "Cross-validating causal discovery via Leave-One-Variable-Out".

## Install
To be able to run the experiments, install all required packages via

    pip install -r requirements.txt

In order to use accelerator hardware for the deep learning model, you might need to do extra steps.
Please refer to the documentation of [JAX](https://jax.readthedocs.io/en/latest/installation.html) for more details.

## Running the experiments
For the simulations with parent adjustment LOVO prediction based on the true marginal graphs, LOVO applied to RCD and DirectLiNGAM, and LOVO via LiNGAM run
    
    python3 LOVO_via_parent_adjustment.py
    python3 LOVO_applied_to_RCD.py
    python3 LOVO_via_LiNGAM.py
    
and to create the plots, use `make_plots.ipynb`.

For the DL experiment, the first step is to train the deep learning model for LOVO prediction.
To this end run

    python3 train_predictor.py

The progress can be visualized via
    
    tensorboard --logdir jax_predictor

The following scipt expects a checkpoint `checkpoint_best` to exist in the main directory.
Pick the training run with which you want to proceed (e.g. `run_42`) and move it
    
    cp -r jax_predictor/run_42/checkpoint_50000 checkpoint_best

Now we are ready to run and visualise the left plot from Figure 3.

    python3 baseline_vs_lingam_vs_dl.py
    python3 plot_pred_error.py
In the current version `LOVO_DL_vs_baseline_vs_LiNGAM.py` does not load the architecture from the checkpoint.
If you change that in `train_predictor.py` you might have to also adjust it in the other file.

To train the model that predicts the causal structure from the embeddings of the former model call

    python3  train_discovery.py

and optionally 

    tensorboard --logdir jax_structure
Again, we need to pick a run that we want to use in the next script.
Additionally, we will move the file where the baseline prediction is stored.
So we run

    cp -r jax_structure/run_42/checkpoint_1000 checkpoint_structure
    cp jax_structure/run_42/baseline_prediction.npy checkpoint_structure

Then we evaluate the result via
    
    python3 test_discovery.py
    python3 plot_hist_structure.py
As before, you might have to change the architecture that is hard coded in `test_discovery.py` if you changed the architecture in `train_discovery.py`.