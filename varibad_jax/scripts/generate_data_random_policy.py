import rlds
import einops
import tqdm
import tensorflow as tf
import numpy as np
from pathlib import Path
from varibad_jax.envs.utils import make_envs, make_procgen_envs, make_atari_envs

if __name__ == "__main__":
    root_dir = "/scr/aliang80/varibad_jax/varibad_jax/"
    save_dir = Path(root_dir) / "tensorflow_datasets" / "procgen_dataset" / "bigfish"
    save_file = save_dir / "random_policy"

    generate = True

    if generate:
        # not jax jit compatible
        num_envs = 1000
        envs = make_procgen_envs(num_envs=num_envs, env_id="coinrun", gamma=1.0)

        obss = []
        actions = []
        rewards = []

        obs = envs.reset()
        obss.append(obs)

        dones = np.bool_(np.zeros(num_envs))

        timesteps = 0

        while not np.all(dones):
            action = np.asarray([envs.action_space.sample() for _ in range(num_envs)])
            obs, reward, dones, info = envs.step(action)

            obss.append(obs)
            actions.append(action)
            rewards.append(reward)

            timesteps += 1

            if timesteps >= 2000:
                break

        envs.close()

        # combine the timesteps together first
        obss = np.stack(obss)
        actions = np.stack(actions)
        rewards = np.stack(rewards)

        # einops to reshape
        obss = einops.rearrange(obss, "t n h w c -> (n t) h w c")
        actions = einops.rearrange(actions, "t n -> (n t)")
        rewards = einops.rearrange(rewards, "t n -> (n t)")

        # # combine the list into a single numpy array
        # if actions.ndim == 1:
        #     actions = actions[:, None]
        obs_shape = obss.shape[1:]
        action_dim = 1

        def generator():
            for obs, action, reward in zip(obss, actions, rewards):
                yield {
                    "observations": obs,
                    "actions": action,
                    "rewards": reward,
                }

        # save to tfds data loader
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature={
                "observations": tf.TensorSpec(shape=obs_shape, dtype=tf.float32),
                "actions": tf.TensorSpec(shape=(), dtype=tf.int32),
                "rewards": tf.TensorSpec(shape=(), dtype=tf.float32),
            },
        )
        # dataset_size = len(dataset)

        # train_size = int(0.8 * dataset_size)
        # val_size = dataset_size - train_size

        # train_ds = dataset.take(train_size)
        # val_ds = dataset.skip(train_size)

        print(f"saving dataset to file: {save_file}")
        # tf.data.experimental.save(train_ds, str(save_file) + "_train")
        # tf.data.experimental.save(val_ds, str(save_file) + "_val")
        tf.data.experimental.save(dataset, str(save_file))
    else:
        # test load
        print(f"loading dataset from file: {save_file}")
        dataset = tf.data.experimental.load(str(save_file))

        dataset = rlds.transformations.batch(dataset, size=2, shift=1)

        import ipdb

        ipdb.set_trace()
        dataset = dataset.shuffle(10000000)
        dataset_size = len(dataset)

        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)

        print(len(train_ds), len(val_ds))

        iterator = train_ds.batch(32).as_numpy_iterator()

        for batch in iterator:
            for key, value in batch.items():
                print(key, value.shape)
            break
