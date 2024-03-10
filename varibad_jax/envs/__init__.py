import gymnasium as gym


def register(id, entry_point, kwargs, force=True):
    env_specs = gym.envs.registry
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(id=id, entry_point=entry_point, kwargs=kwargs)


register(
    "GridNavi-v0",
    entry_point=("varibad_jax.envs.gridworld:GridNavi"),
    kwargs={},
)
register(
    "GridNaviJAX-v0",
    entry_point=("varibad_jax.envs.gridworld_jax:GridNavi"),
    kwargs={},
)
