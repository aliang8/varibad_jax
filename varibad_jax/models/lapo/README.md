# Reimplementation of LAPO

## Architecture:
- ### Latent Inverse Dynamics Model (IDM) `model.py`
    - predicts the latent action $z_t$ given $o_{t-1}$, $o_t$ and $o_{t+1}$
    - $p_{IDM}(z_t \mid o_{t-k}, \dots, o_t, o_t+1)$
    - ImpalaCNN encodes the observation $N,(3 * C), H, W \rightarrow N, D$ where $D$ is the latent action dimension
    - Apply VQEMA $N,D \rightarrow N,D$
- ### Latent Forward Dynamics Model (FDM) `model.py`
    - predicts $o_{t+1}$ given $o_{t-1}$, $o_t$ and $z_t$ where $z_t$ is the latent action inferred by the IDM
    - $p_{FDM}(o_{t+1} \mid o_{t-k}, \dots, o_t, z_t)$
    - Implemented as a U-net style architecture (CNN encoders and decoders)
    - Input is $N,T,C,H,W$ where $T=2$ because $o_{t-1}$ and $o_t$
    - Conatenate the latent action with context, $N,(T*C+D_L),H,W$ where $D_L$ is the latent action dim
    - Apply downsampling conv layers, $N,(T*C+D_L),H,W \rightarrow N,32,H_{CONV},W_{CONV}$
    - LAPO does a thing where they also inject the latent action into the middle of the U-net
- ### BC Policy
    - MLP with 2 FC layers, 128 dimensions each, predicts the latent action
    - $p_{BC}(z_t | o_t)$ 
    - Minimize MSE between predicted latent actions and latent action from IDM
- ### Latent Action Decoder 
    - MLP with 2 FC layers, 128 dimensions each
    - $p_{dec}(a_t | z_t)$

## Notes
- LAPO uses one previous observation as context ($k=1$)
- Training objective: minimize reconstruction error ($||\hat{o_{t+1}} - o_{t+1}||$)
- Only supports training with XLand offline dataset (observations are 5x5x2 images)
    - Modify the CNN encoder in the IDM and FDM
    - Currently training w/ 10k trajectories collected using a pretrained VariBAD policy
- Use labelled data with randomly selected 500 trajectories for training the action decoder
- Hyperparameters I am not sure about: VQVAE, CNN architecture
- TODO: 
    - [x] Implement latent BC policy ($\mathcal{O} \rightarrow \mathcal{Z}$)
    - [ ] Implement decoder to predict ground truth actions from latent actions