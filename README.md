## JAX Implementations of Meta-RL / Offline-RL algorithms

This repository provides clean reimplementations of existing Meta-RL and Offline-RL algorithms. 

### Getting started:
```
conda create --name jax_metarl python==3.11.8
pip3 install -e . # should install this repo and dependencies
```

### Example command 
Run basic goal-conditioned RL 
```
python3 main.py \
    --config=configs/rl_config.py:lstm-gridworld \
    --config.smoke_test=True \
    --config.use_wb=False
```

Run VariBAD on XLand using LSTM encoder
```
# gridworld 
python3 main.py \
    --config=configs/varibad_config.py:lstm-gridworld \
    --config.smoke_test=True \
    --config.use_wb=False \
    --config.overwrite=False

# XLand
CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --config=configs/varibad_config.py:lstm-xland-5x5 \
    --config.smoke_test=True \
    --config.use_wb=False
```

Collect offline dataset with trained model
```
python3 scripts/generate_data_from_policy.py \
    --config=configs/offline_config.py:gridworld \
    --config.model_ckpt_dir=results/en-gridworld_alg-ppo_pltp-True_t-vae_nvu-3_ed-8 \

CUDA_VISIBLE_DEVICES=0 python3 scripts/generate_data_from_policy.py \
    --config=configs/offline_config.py:xland-5x5 \
    --config.model_ckpt_dir=results/en-xland_alg-ppo_pltp-True_t-vae_nvu-3_ed-8

python3 scripts/generate_data_from_policy.py \
    --config=configs/offline_config.py:dt-xland-5x5 \
    --config.model_ckpt_dir=results/ \
```

Run offline RL experiments with Decision Transformer
```
CUDA_VISIBLE_DEVICES=4 python3 main.py \
    --config=configs/offline_config.py:dt-gridworld \
    --config.smoke_test=True \
    --config.use_wb=False
```

```
Train LAPO
CUDA_VISIBLE_DEVICES=4 python3 main.py \
    --config=configs/offline_config.py:lapo-xland-5x5 \
    --config.smoke_test=True \
    --config.use_wb=False
```


Also supports using Ray for hyperparameter search and WandB for logging experiment metrics. Use `smoke_test` to toggle Ray tune. 

### File organization:


Meta-RL algorithms supported:
- [x] VariBAD
- [ ] RL^2
- [x] HyperX, not working yet

Offline RL:
- [x] Decision Transformers 
- [ ] LAPO

Misc Models:
- [ ] Genie

Environments supported:
- [x] Gridworld
- [x] Xland-Minigrid 
- [ ] DM-Alchemy
- [ ] ProcGen

