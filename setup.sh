conda create -y --prefix ./env python=3.9
conda activate ./env
pip install -e .


# the order of installing torch and jax matters
# install torch
pip install torch 

# install jax
# set cuda correctly in LD_LIBRARY_PATH, might conflict with the torch version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
