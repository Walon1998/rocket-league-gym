from abc import abstractmethod

import numpy as np

from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym.utils.gamestates import PlayerData, GameState

from numba import int32, float32, boolean, float64, types, typed, typeof  # import the types
from numba.experimental import jitclass
import rlgym.utils.reward_functions.common_rewards as bb


class ConditionalRewardFunction(RewardFunction):
    def __init__(self, reward_func: RewardFunction):
        super().__init__()
        self.reward_func = reward_func

    @abstractmethod
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_reward(player, state, previous_action)
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_final_reward(player, state, previous_action)
        return 0


# spec_constant = [
#     ('reward_func', bb.ConstantReward.class_type.instance_type),
#     ('team_only', boolean),
# ]


# @jitclass(spec_constant)
class RewardIfClosestToBall(object):
    def __init__(self, reward_func: RewardFunction, team_only=True):
        self.reward_func = reward_func
        self.team_only = team_only

    def pre_step(self, initial_state):
        pass

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        dist = np.linalg.norm(player.car_data.position - state.ball.position)
        for player2 in state.players:
            if not self.team_only or player2.team_num == player.team_num:
                dist2 = np.linalg.norm(player2.car_data.position - state.ball.position)
                if dist2 < dist:
                    return False
        return True


    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_reward(player, state, previous_action)
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_final_reward(player, state, previous_action)
        return 0

# @jitclass(spec_constant)
class RewardIfTouchedLast(object):
    def __init__(self, reward_func):
        self.reward_func = reward_func

    def pre_step(self, initial_state):
        pass
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return state.last_touch == player.car_id

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_reward(player, state, previous_action)
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_final_reward(player, state, previous_action)
        return 0


# @jitclass(spec_constant)
class RewardIfBehindBall(object):
    def __init__(self, reward_func):
        self.reward_func = reward_func

    def pre_step(self, initial_state):
        pass

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return player.team_num == BLUE_TEAM and player.car_data.position[1] < state.ball.position[1] \
               or player.team_num == ORANGE_TEAM and player.car_data.position[1] > state.ball.position[1]

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_reward(player, state, previous_action)
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_final_reward(player, state, previous_action)
        return 0
