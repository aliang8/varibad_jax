## JAX Implementations of Meta-RL / Offline-RL algorithms

This repository provides clean reimplementations of existing Meta-RL and Offline-RL algorithms. 

### Getting started:
```
conda env create --name jax_metarl python==3.11.8
pip3 install -e . # should install this repo and dependencies
```

### Example command 
Run VariBAD on XLand using LSTM encoder
```
python3 main.py \
    --config=configs/varibad_config.py:lstm-xland \
    --config.smoke_test=True \
    --config.use_wb=True
```

Also supports using Ray for hyperparameter search and WandB for logging experiment metrics. Use `smoke_test` to toggle Ray tune. 

### File organization:


Meta-RL algorithms supported:
- [x] VariBAD
- [ ] RL^2
- [ ] HyperX

Offline RL:
- [x] Decision Transformers 

Misc Models:
- [ ] Genie

Environments supported:
- [x] Gridworld
- [x] Xland-Minigrid 
- [ ] DM-Alchemy

