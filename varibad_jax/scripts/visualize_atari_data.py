import cv2
import numpy as np
import matplotlib.pyplot as plt
import jax
import tensorflow as tf
from ml_collections.config_dict import ConfigDict
from varibad_jax.envs.atari import create_atari_env
from varibad_jax.envs.utils import make_atari_envs
from varibad_jax.utils.tfds_data_utils import load_data

action_map = {
    0: "NOOP",
    1: "FIRE",
    2: "LEFT",
    3: "RIGHT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}

if __name__ == "__main__":
    config = ConfigDict(
        {
            "root_dir": "/scr/aliang80/varibad_jax/varibad_jax",
            "env": {"env_name": "atari", "env_id": "Breakout"},
            "data": {
                "num_trajs": 100,
                "batch_size": 32,
                "context_len": 2,
                "num_frame_stack": 1,
                "data_type": "trajectories",
                "load_latent_actions": False,
                "dataset_name": "rlu_atari_checkpoints_ordered",
            },
        }
    )
    rng = jax.random.PRNGKey(0)
    tf.random.set_seed(1)

    # env = create_atari_env("Pong")
    # obs1 = np.array(env.reset().observation)[..., -1] / 255.0

    # env2 = make_atari_envs(1, "Pong")
    # obs2, _ = env2.reset()
    # obs2 = obs2[-1] / 255.0

    train_ds, val_ds, test_ds = load_data(config, rng)
    fps = 5
    width, height = 84, 84

    for batch in train_ds:
        observations = batch["observations"].numpy()
        actions = batch["actions"].numpy()

        print(observations.shape, actions.shape)
        observations = observations[0]

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(16):
            frame = observations[i + 80, ..., 0]
            action = actions[0][i + 80].item()
            action_str = action_map[action]
            # add text
            frame = cv2.putText(
                frame,
                f"Action: {action_str}",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.2,
                (240, 128, 128),
                1,
                cv2.LINE_AA,
            )
            frame = frame.squeeze()
            axes[i].imshow(frame, cmap="gray")
            axes[i].axis("off")

        plt.savefig(f"frames_{config.env.env_id}.png")
        break

        # # frames = observations[0]
        # # frames = frames.repeat(3, -1)

        # # print(frames.shape)
        # # print(frames.min(), frames.max())

        # # single_frame = observations[0, 0, ..., -1]
        # single_frame = observations[0, ..., 0]
        # print(single_frame.shape)

        # print((single_frame == obs1).all(), (single_frame == obs1).sum())
        # print((single_frame == obs2).all(), (single_frame == obs2).sum())

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(single_frame, cmap="gray")
        # ax[1].imshow(obs1, cmap="gray")
        # # ax[2].imshow(obs2, cmap="gray")
        # ax[2].imshow(single_frame - obs2, cmap="gray")
        # plt.savefig("frames.png")

        # import ipdb

        # ipdb.set_trace()

        # # make a video out of the frames
        # out = cv2.VideoWriter(
        #     "video.mp4",
        #     cv2.VideoWriter_fourcc(*"mp4v"),
        #     fps,
        #     (width, height),
        #     # isColor=False,
        # )

        # for i in range(len(frames)):
        #     data = frames[i]
        #     action = actions[0][i]

        #     data = cv2.putText(
        #         data,
        #         f"Action: {action}",
        #         (0, 20),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.3,
        #         (240, 128, 128),
        #         1,
        #         cv2.LINE_AA,
        #     )
        #     out.write(data)

        # out.release()

        # break
