"""
definition of requests for the Ricardo's Simulator
"""

from src.simulator.Simulator_platform import *
from src.simulator.route_functions import *

class Req(object):
    """
    Req is a class for requests
    Attributes:
    
        Ori_id: id of Origin node
        Des_id: id of Destination node
        Req_time: request time
        Num_people: number of people in the request
        Max_wait: max waiting time of this request
        Req_ID: ID of the request
        Latest_PU_Time: latest pickup time
        Latest_DO_Time: latest dropoff time
        Shortest_TT: shortest travel time
        # Shortest_TD: shortest travel distance
        Actual_PU_Time: actual pickup time
        Actual_DO_Time: actual dropoff time

        Status: Pending/Onboard/Complete

        ---Not in Use---
        D: detour factor
        
    """
    # Small Grid Data
    # def __init__(self, Ori_id: int, Des_id: int, Req_time: float, Num_people: int, Max_wait: float, Req_ID: int):
    #     self.Ori_id = Ori_id
    #     self.Des_id = Des_id
    #     self.Req_time = Req_time
    #     self.Num_people = Num_people
    #     self.Max_wait = Max_wait
    #     self.Req_ID = Req_ID
    #     self.Shortest_TT = get_timeCost(self.Ori_id, self.Des_id)
    #     self.Latest_PU_Time = self.Req_time + self.Max_wait
    #     self.Latest_DO_Time = self.Req_time + self.Shortest_TT + Max_wait * 2
    #     self.Actual_PU_Time = -1.0
    #     self.Actual_DO_Time = -1.0
    #     self.Status = OrderStatus.PENDING

    # Manhattan Data
    def __init__(self, Req_ID: int, Ori_id: int, Des_id: int, Req_time: int, Num_people: int):
        self.Ori_id = Ori_id
        self.Des_id = Des_id
        self.Req_time = Req_time
        self.Num_people = Num_people
        self.Req_ID = Req_ID
        self.Shortest_TT = get_timeCost(self.Ori_id, self.Des_id)
        self.Latest_PU_Time = self.Req_time + MAX_PICKUP_WAIT_TIME
        self.Latest_DO_Time = self.Req_time + MAX_PICKUP_WAIT_TIME + self.Shortest_TT + MAX_DETOUR_TIME 
        self.Status = OrderStatus.PENDING

    def update_pick_info(self, system_time: float):
        self.Actual_PU_Time = system_time
        # #  DEBUG codes
        # if self.status != OrderStatus.PICKING:
        #     print(f"[DEBUG1] req {self.id}, {self.status}, "
        #           f"request time {self.Tr}, latest pickup {self.Clp}, pickup time {self.Tp}")
        # assert (self.status == OrderStatus.PICKING)
        self.status = OrderStatus.ONBOARD

    def update_drop_info(self, system_time: float):
        self.Actual_DO_Time = system_time
        # self.D = (self.Td - self.Tp) / self.Ts
        assert(self.status == OrderStatus.ONBOARD)
        self.status = OrderStatus.COMPLETE
