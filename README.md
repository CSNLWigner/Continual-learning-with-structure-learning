# Continual-learning-with-structure-learning

The goal of this project is to build a toy model for continual learning with hierarchical generative model.

## Files

### Py files
- helper: plotting and data dictionary related functions
- helper_mate: plotting and data dictionary related functions, Mate's version
- colorscheme: definition of color schemes for plots
- gt_learner:
- gr_em_learner:
- models: 

### Notebooks
- exp_GT_cardinal: a single run of a GT learner on cardinal data
- exp_GT_diagonal: a single run of a GT learner diagonal data
- exp_GT_cardinal_batch: batch run of GT learner on cardinal data, switching times statistics
- exp_GT_data_rotation: experiment for effect of task complexity, computes model switching times on original and rotated data
- exp_GR_cardinal: scatter plot of GT vs GR switching times for cardinal data
- exp_EMvsGR_cardinal: scatter plot of GT vs GR vs EM switching times for cardinal data

- experiments_1: Gergo's experiments


## Requirements

### macOS

```bash
conda create -n tf2 tensorflow
conda activate tf2
pip install tensorflow=2.10 tensorflow-probability==0.18 matplotlib pandas
```
