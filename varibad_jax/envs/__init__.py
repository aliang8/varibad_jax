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

from xminigrid.registration import register as register_xminigrid

register_xminigrid(
    id="XLand-MiniGridCustom-R1-7x7",
    entry_point="varibad_jax.envs.xland.custom:CustomXLandMiniGrid",
    grid_type="R1",
    height=7,
    width=7,
    view_size=5,
)

register_xminigrid(
    id="XLand-MiniGridCustom-R1-7x7-3",
    entry_point="varibad_jax.envs.xland.custom:CustomXLandMiniGrid",
    grid_type="R1",
    height=7,
    width=7,
    view_size=3,
)

register_xminigrid(
    id="XLand-MiniGridCustom-R1-9x9-3",
    entry_point="varibad_jax.envs.xland.custom:CustomXLandMiniGrid",
    grid_type="R1",
    height=9,
    width=9,
    view_size=3,
)


register_xminigrid(
    id="XLand-MiniGrid-TwoGoals-R1-7x7-5",
    entry_point="varibad_jax.envs.xland.two_goals:TwoGoals",
    grid_type="R1",
    height=7,
    width=7,
    view_size=5,
)

register_xminigrid(
    id="MiniGrid-GoToDoor-R1-5x5",
    entry_point="varibad_jax.envs.xland.go_to_door:GoToDoor",
    height=5,
    width=5,
    view_size=5,
)


register_xminigrid(
    id="MiniGrid-GoToDoor-R1-7x7",
    entry_point="varibad_jax.envs.xland.go_to_door:GoToDoor",
    height=7,
    width=7,
    view_size=5,
)

register_xminigrid(
    id="MiniGrid-GoToDoor-R1-9x9",
    entry_point="varibad_jax.envs.xland.go_to_door:GoToDoor",
    height=9,
    width=9,
    view_size=5,
)


# register_xminigrid(
#     id="XLand-MiniGrid-R2-9x9",
#     entry_point="xminigrid.envs.xland:XLandMiniGrid",
#     grid_type="R2",
#     height=9,
#     width=9,
#     view_size=5,
# )
