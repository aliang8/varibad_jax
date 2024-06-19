import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for i in range(100):
        # load npz file
        file = f"/scr/aliang80/varibad_jax/varibad_jax/datasets/procgen/expert_data/coinrun/train/trajs/traj_{i}.npz"

        # load
        data = np.load(file, allow_pickle=True)
        num_timesteps = len(data["observations"])
        print(f"num_timesteps: {num_timesteps}")
        print(data["actions"])

        print(data["observations"].shape)

        frame = data["observations"][0]
        last_frame = data["observations"][-1]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(frame)
        ax[1].imshow(last_frame)
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.savefig(f"frame_{i}.png")
        break
