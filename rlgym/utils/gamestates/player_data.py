"""
A class containing all data about a player in the game.
"""

from rlgym.utils.gamestates import PhysicsObject

from numba import int32, float32, boolean, float64  # import the types
from numba.experimental import jitclass

spec_PlayerData = [
    ('car_id', int32),
    ('team_num', int32),
    ('match_goals', int32),
    ('match_saves', int32),
    ('match_shots', int32),
    ('match_demolishes', int32),
    ('boost_pickups', int32),
    ('is_demoed', boolean),
    ('on_ground', boolean),
    ('ball_touched', boolean),
    ('has_jump', boolean),
    ('has_flip', boolean),
    ('boost_amount', float32),
    ('car_data', PhysicsObject.class_type.instance_type),
    ('inverted_car_data', PhysicsObject.class_type.instance_type),
]

@jitclass(spec_PlayerData)
class PlayerData(object):
    def __init__(self):
        self.car_id: int = -1
        self.team_num: int = -1
        self.match_goals: int = -1
        self.match_saves: int = -1
        self.match_shots: int = -1
        self.match_demolishes: int = -1
        self.boost_pickups: int = -1
        self.is_demoed: bool = False
        self.on_ground: bool = False
        self.ball_touched: bool = False
        self.has_jump: bool = False
        self.has_flip: bool = False
        self.boost_amount: float = -1
        self.car_data: PhysicsObject = PhysicsObject(None, None, None, None)
        self.inverted_car_data: PhysicsObject = PhysicsObject(None, None, None, None)

    # def __str__(self):
    #     output = "****PLAYER DATA OBJECT****\n" \
    #              "Match Goals: {}\n" \
    #              "Match Saves: {}\n" \
    #              "Match Shots: {}\n" \
    #              "Match Demolishes: {}\n" \
    #              "Boost Pickups: {}\n" \
    #              "Is Alive: {}\n" \
    #              "On Ground: {}\n" \
    #              "Ball Touched: {}\n" \
    #              "Has Jump: {}\n" \
    #              "Has Flip: {}\n" \
    #              "Boost Amount: {}\n" \
    #              "Car Data: {}\n" \
    #              "Inverted Car Data: {}"\
    #         .format(self.match_goals,
    #                 self.match_saves,
    #                 self.match_shots,
    #                 self.match_demolishes,
    #                 self.boost_pickups,
    #                 not self.is_demoed,
    #                 self.on_ground,
    #                 self.ball_touched,
    #                 self.has_jump,
    #                 self.has_flip,
    #                 self.boost_amount,
    #                 self.car_data,
    #                 self.inverted_car_data)
    #     return output
