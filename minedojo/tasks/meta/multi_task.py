from typing import List, Optional, Union, Tuple, Dict
from functools import partial

from .base import ExtraSpawnMetaTaskBase
from ...sim.inventory import InventoryItem
from .extra_spawn import SpawnItem2Condition
from .utils import (
    always_satisfy_condition,
    empty_check,
    empty_reward,
    simple_stat_kill_entity_based_check,
    simple_stat_kill_entity_based_reward,
    simple_inventory_based_check,
    simple_inventory_based_reward,
    check_success_base,
    reward_fn_base,
)

COMBAT_TARGET = [
    "cow",
    "sheep",
    "chicken",
    "pig",
    "horse",
    "llama",
    "wolf",
    "donkey",
    "bat",
    "spider",
    "rabbit",
    # "mooshroom",
    "mushroom_cow",
    "zombie",
    "pig_zombie",
    "skeleton",
    "creeper",
    "enderman",
]

HARVEST_TARGET = [
    "log",
    "leaves",
    "dirt",
    "sand",
    "milk",
    "cow_milk",
    "mooshroom_milk",
    "wool",
    # flowers
    "flower",
    "dandelion",
    "poppy",
    "allium",
    "azure_bluet",
    "red_tulip",
    "orange_tulip",
    "white_tulip",
    "pink_tulip",
    "oxeye_daisy",
    "sunflower",
    "lilac",
    "rose",
    "grass",
    "water",
    "redstone_torch",
    "torch",
    "mushroom",
    "stone",
]

# TARGET with different name for VLM
HARVEST_TARGET2VLM = {
    "log": "oak wood or birch wood",
    "leaves": "oak leaves or birch leaves",
    "milk": "cow",
    "cow_milk": "cow",
    "mooshroom_milk": "mooshroom",
    "wool": "sheep",
    "azure_bluet": "azure bluet",
    "red_tulip": "red tulip",
    "orange_tulip": "orange tulip",
    "white_tulip": "white tulip",
    "pink_tulip": "pink tulip",
    "oxeye_daisy": "oxeye daisy",
    "rose": "rose bush",
    "redstone_torch": "redstone torch",
    "water": "lake",
    "mushroom": "mooshroom",
}

HARVEST_TARGET2MCID = {
    "log": "log",
    "leaves": "leaves",
    "dirt": "dirt",
    "sand": "sand",
    "milk": "milk_bucket",
    "cow_milk": "milk_bucket",
    "mooshroom_milk": "milk_bucket",
    "wool": "wool",
    "dandelion": "yellow_flower",
    "flower": "red_flower",
    "poppy": "red_flower",
    "allium": "red_flower",
    "azure_bluet": "red_flower",
    "red_tulip": "red_flower",
    "orange_tulip": "red_flower",
    "white_tulip": "red_flower",
    "pink_tulip": "red_flower",
    "oxeye_daisy": "red_flower",
    "sunflower": "double_plant",
    "lilac": "double_plant",
    "rose": "double_plant",
    "grass": "tallgrass",
    "water": "water_bucket",
    "redstone_torch": "redstone_torch",
    "torch": "torch",
    "mushroom": "red_mushroom",
    "stone": "cobblestone",
}

HARVEST_TARGET2VARIANT = {
    "poppy": 0,
    "allium": 2,
    "azure_bluet": 3,
    "red_tulip": 4,
    "orange_tulip": 5,
    "white_tulip": 6,
    "pink_tulip": 7,
    "oxeye_daisy": 8,
    "sunflower": 0,
    "lilac": 1,
    "rose": 4,
}

def _variant_inventory_based_check(
    name: str,
    variant: int,
    quantity: int,
    ini_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple success check based on `info["inventory"]`
    """
    return (
        sum(
            [
                inv_item["quantity"]
                for inv_item in cur_info_dict["inventory"]
                if inv_item["name"] == name and inv_item["variant"] == variant
            ]
        )
        - sum(
            [
                inv_item["quantity"]
                for inv_item in ini_info_dict["inventory"]
                if inv_item["name"] == name and inv_item["variant"] == variant
            ]
        )
    ) >= quantity


def variant_inventory_based_check(
    name: str, variant: int , quantity: int, **kwargs
) -> check_success_base:
    return partial(_variant_inventory_based_check, name=name, variant=variant, quantity=quantity)


def _variant_inventory_based_reward(
    name: str,
    variant: int,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["inventory"]`
    """
    return (
        sum(
            [
                inv_item["quantity"]
                for inv_item in cur_info_dict["inventory"]
                if inv_item["name"] == name and inv_item["variant"] == variant
            ]
        )
        - sum(
            [
                inv_item["quantity"]
                for inv_item in pre_info_dict["inventory"]
                if inv_item["name"] == name and inv_item["variant"] == variant
            ]
        )
    ) * weight


def variant_inventory_based_reward(
    name: str, variant: int, weight: Union[int, float], **kwargs
) -> reward_fn_base:
    return partial(_variant_inventory_based_reward, name=name, variant=variant, weight=weight)


class MultiTaskMeta(ExtraSpawnMetaTaskBase):
    """
    Class for combat tasks.
    Args:
        allow_mob_spawn: If ``True``, allow mobs (animals and hostiles) to spawn.
                Default: ``True``.

        always_night: If ``True``, the world will always be at night.
                Default: ``False``.

        break_speed_multiplier: Controls the speed of breaking blocks. A value larger than 1.0 accelerates the breaking.
                Default: ``1.0``.

        drawing_str: Draws shapes (e.g. spheres, cuboids) in the minecraft world.
                Default: ``None``.

        event_level_control: If ``True``, the agent is able to perform high-level controls including place and equip.
                If ``False``, then is keyboard-mouse level control.
                Default: ``True``.

        fast_reset: If ``True``, reset using MC native command `/kill`, instead of waiting for re-generating new worlds.
            Default: ``True``.

            .. warning::
                Side effects:

                1. Changes to the world will not be reset. E.g., if the agent chops lots of trees then calling
                fast reset will not restore those trees.

                2. If you specify agent starting health and food, these specs will only be respected at the first reset
                (i.e., generating a new world) but will not be respected in the following resets (i.e., reset using MC cmds).
                So be careful to use this wrapper if your usages require specific initial health/food.

                3. Statistics/achievements will not be reset. This wrapper will maintain a property ``info_prev_reset``.
                If your tasks use stat/achievements to evaluation, please retrieve this property and compute differences.

        image_size: The size of image observations.

        initial_inventory: If not ``None``, specifies initial items in the agent's inventory.
                Use class ``InventoryItem`` to specify items.
                Default: ``None``.

        initial_mobs: The types of mobs that are spawned initially.
                Default: ``None``.

        initial_mob_spawn_range_high: The upper bound on each horizontal axis from the center of the area to spawn initially.
                Default: ``None``.

        initial_mob_spawn_range_low: The lower bound on each horizontal axis from the center of the area to spawn initially.
                Default: ``None``.

        initial_weather: If not ``None``, specifies the initial weather.
                Can be one of ``"clear"``, ``"normal"``, ``"rain"``, ``"thunder"``.
                Default: ``None``.

        lidar_rays: Defines the directions and maximum distances of the lidar rays if ``use_lidar`` is ``True``.
                If supplied, should be a list of tuple(pitch, yaw, distance).
                Pitch and yaw are in radians and relative to agent looking vector.
                Default: ``None``.

        reward_weights: The reward weight for each target in the task.
                Default: ``1.0``.

        seed: The seed for an instance's internal generator.
                Default: ``None``.

        sim_name: Name of a simulation instance.
                Default: ``"CombatMeta"``.

        spawn_range_high: The upper bound on each horizontal axis from the center of the area to spawn
                Default: ``None``.

        spawn_range_low: The lower bound on each horizontal axis from the center of the area to spawn
                Default: ``None``.

        spawn_rate: The probability of spawning in each step
                Default: ``None``.

        specified_biome: The specified biome of the task.
                Default: ``None``.

        start_at_night: If ``True``, the task starts at night.
                Default: ``True``.

        start_food: If not ``None``, specifies initial food condition of the agent.
                Default: ``None``.

        start_health: If not ``None``, specifies initial health condition of the agent.
                Default: ``None``.

        start_position: If not ``None``, specifies the agent's initial location and orientation in the minecraft world.
                Default: ``None``.

        use_lidar: If ``True``, includes lidar in observations.
                Default: ``False``.

        use_voxel: If ``True``, includes voxel in observations.
                Default: ``False``.

        voxel_size: Defines the voxel's range in each axis if ``use_voxel`` is ``True``.
                If supplied, should be a dict with keys ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``.
                Each value specifies the voxel size relative to the agent.
                Default: ``None``.

        world_seed: Seed for deterministic world generation
                if ``generate_world_type`` is ``"default"`` or ``"specified_biome"``.
                See `here <https://minecraft.fandom.com/wiki/Seed_(level_generation)>`_ for more details.
                Default: ``None``.
    """

    _prompt_template = "Hunt {targets}."

    def __init__(
        self,
        *,
        # ------ hunting targets, quantities, and reward weights ------
        target_names: Union[str, List[str]],
        target_quantities: Union[int, List[int], Dict[str, int]],
        reward_weights: Union[int, float, Dict[str, Union[int, float]]] = 1.0,
        # ------ initial & extra spawn control ------
        initial_mobs: Optional[Union[str, List[str]]] = None,
        initial_mob_spawn_range_low: Optional[Tuple[int, int, int]] = None,
        initial_mob_spawn_range_high: Optional[Tuple[int, int, int]] = None,
        spawn_rate: Optional[
            Union[float, int, List[Union[float, int]], Dict[str, Union[float, int]]]
        ] = None,
        spawn_range_low: Optional[Tuple[int, int, int]] = None,
        spawn_range_high: Optional[Tuple[int, int, int]] = None,
        # ------ initial conditions ------
        start_position: Optional[Dict[str, Union[float, int]]] = None,
        start_health: Optional[float] = None,
        start_food: Optional[int] = None,
        start_at_night: bool = True,
        initial_weather: Optional[str] = None,
        initial_inventory: Optional[List[InventoryItem]] = None,
        drawing_str: Optional[str] = None,
        # ------ global conditions ------
        specified_biome: Optional[Union[int, str]] = None,
        always_night: bool = False,
        allow_mob_spawn: bool = True,
        break_speed_multiplier: float = 1.0,
        # ------ sim seed & world seed ------
        seed: Optional[int] = None,
        world_seed: Optional[str] = None,
        # ------ reset mode ------
        fast_reset: bool = True,
        # (fast_reset_range --> ~_high & ~_low)
        fast_reset_random_teleport_range: Optional[int] = None,
        fast_reset_random_teleport_range_high: Optional[int] = None,
        fast_reset_random_teleport_range_low: Optional[int] = None,
        # ------ obs ------
        image_size: Union[int, Tuple[int, int]],
        use_voxel: bool = False,
        voxel_size: Optional[Dict[str, int]] = None,
        use_lidar: bool = False,
        lidar_rays: Optional[List[Tuple[float, float, float]]] = None,
        # ------ event-level action or keyboard-mouse level action ------
        event_level_control: bool = True,
        # ------ misc ------
        sim_name: str = "MultiTaskMeta",
    ):
        if isinstance(target_names, str):
            target_names = [target_names]
        if isinstance(target_quantities, int):
            target_quantities = {k: target_quantities for k in target_names}
        elif isinstance(target_quantities, list):
            assert len(target_names) == len(target_quantities)
            target_quantities = {
                k: target_quantities[i] for i, k in enumerate(target_names)
            }
        elif isinstance(target_quantities, dict):
            assert set(target_names) == set(target_quantities.keys())
        if isinstance(reward_weights, int) or isinstance(reward_weights, float):
            reward_weights = {k: reward_weights for k in target_names}
        elif isinstance(reward_weights, dict):
            assert set(target_names) == set(reward_weights.keys())
        self._target_quantities = target_quantities

        spawn_condition = None
        if spawn_rate is not None:
            # TODO: implement Target2SpawnItem for harvest target
            if isinstance(spawn_rate, float) or isinstance(spawn_rate, int):
                spawn_rate = {k: spawn_rate for k in target_names}
            elif isinstance(spawn_rate, list):
                assert len(spawn_rate) == len(target_names)
                spawn_rate = {k: spawn_rate[i] for i, k in enumerate(target_names)}
            elif isinstance(spawn_rate, dict):
                # don't do any checks here, so users can specify arbitrary spawn rates
                pass
            spawn_condition = {
                k: SpawnItem2Condition.get(k, always_satisfy_condition)
                for k in spawn_rate.keys()
            }

        success_criteria = []
        for k, v in target_quantities.items():
            if k in COMBAT_TARGET:
                # special case [temporal]
                if k == "mooshroom":
                    k_ = "mushroom_cow"
                else:
                    k_ = k
                success_criteria.append(
                    simple_stat_kill_entity_based_check(name=k, quantity=v)
                )
            elif k in HARVEST_TARGET:
                if k in HARVEST_TARGET2VARIANT:
                    success_criteria.append(
                        variant_inventory_based_check(
                            name=HARVEST_TARGET2MCID[k],
                            variant=HARVEST_TARGET2VARIANT[k],
                            quantity=v,
                        )
                    )
                else:
                    success_criteria.append(
                        simple_inventory_based_check(
                            name=HARVEST_TARGET2MCID[k],
                            quantity=v,
                        )
                    )
            else:
                # raise ValueError(f"Unknown target name {k}")
                print(f"Warning: unknown target name {k}, empty success criteria added.")
                success_criteria.append(empty_check())

        reward_fns = []
        for k, v in reward_weights.items():
            if k in COMBAT_TARGET:
                if k == "mooshroom":
                    k_ = "mushroom_cow"
                else:
                    k_ = k
                reward_fns.append(
                    simple_stat_kill_entity_based_reward(name=k_, weight=v)
                )
            elif k in HARVEST_TARGET:
                if k in HARVEST_TARGET2VARIANT:
                    reward_fns.append(
                        variant_inventory_based_reward(
                            name=HARVEST_TARGET2MCID[k],
                            variant=HARVEST_TARGET2VARIANT[k],
                            weight=v,
                        )
                    )
                else:
                    reward_fns.append(
                        simple_inventory_based_reward(
                            name=HARVEST_TARGET2MCID[k], 
                            weight=v,
                        )
                    )
            else:
                # raise ValueError(f"Unknown target name {k}")
                print(f"Warning: unknown target name {k}, no reward function added.")
                reward_fns.append(empty_reward())

        # success_criteria = [
        #     simple_stat_kill_entity_based_check(name=k, quantity=v)
        #     for k, v in target_quantities.items()
        # ]
        # reward_fns = [
        #     simple_stat_kill_entity_based_reward(name=k, weight=v)
        #     for k, v in reward_weights.items()
        # ]

        start_time, allow_time_passage = None, True
        if start_at_night:
            start_time = 13000
        if always_night:
            start_time = 18000
            allow_time_passage = False

        # generate_world_type = (
        #     "default" if specified_biome is None else "specified_biome"
        # )
        if specified_biome is None:
            generate_world_type = "default"
        elif specified_biome == "flat":
            generate_world_type = "flat"
            specified_biome = None
        else:
            generate_world_type = "specified_biome"

        super().__init__(
            initial_mobs=initial_mobs,
            initial_mob_spawn_range_low=initial_mob_spawn_range_low,
            initial_mob_spawn_range_high=initial_mob_spawn_range_high,
            extra_spawn_rate=spawn_rate,
            extra_spawn_condition=spawn_condition,
            extra_spawn_range_low=spawn_range_low,
            extra_spawn_range_high=spawn_range_high,
            fast_reset=fast_reset,
            fast_reset_random_teleport_range=fast_reset_random_teleport_range,
            fast_reset_random_teleport_range_high=fast_reset_random_teleport_range_high,
            fast_reset_random_teleport_range_low=fast_reset_random_teleport_range_low,
            success_criteria=success_criteria,
            reward_fns=reward_fns,
            seed=seed,
            sim_name=sim_name,
            image_size=image_size,
            use_voxel=use_voxel,
            voxel_size=voxel_size,
            use_lidar=use_lidar,
            lidar_rays=lidar_rays,
            event_level_control=event_level_control,
            initial_inventory=initial_inventory,
            break_speed_multiplier=break_speed_multiplier,
            world_seed=world_seed,
            start_position=start_position,
            initial_weather=initial_weather,
            start_time=start_time,
            allow_time_passage=allow_time_passage,
            start_health=start_health,
            start_food=start_food,
            drawing_str=drawing_str,
            allow_mob_spawn=allow_mob_spawn,
            generate_world_type=generate_world_type,
            specified_biome=specified_biome,
        )

    @property
    def task_prompt(self) -> str:
        filling = ", ".join(
            [
                f"{v} {str(k).replace('_', ' ')}"
                for k, v in self._target_quantities.items()
            ]
        )
        return super().get_prompt(targets=filling)
