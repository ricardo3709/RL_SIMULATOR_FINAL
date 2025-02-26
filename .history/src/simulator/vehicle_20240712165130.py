"""
definition of vehicles for the AMoD system
"""

from src.simulator.route_functions import *
from src.simulator.Simulator_platform import *



class Veh(object):
    """
    Veh is a class for vehicles
    Attributes:
        id: vehicle ID
        status: idle/working/rebalancing
        veh_time: time recording by veh 
        arc_completion_percentage: percentage of the current arc that has been completed
        time_to_complete_current_arc: time needed to complete the current arc
        current_node: id of current node in network (last node visited by the vehicle)
        target_node: id of next node the vehicle will visit (route[0])
        route: a list of nodes that the vehicle will visit
        capacity: capacity of the vehicle
        load: number of passengers on board
        schedule: a list of nodes that the vehicle plan to visit(planned PU/DO nodes, includes visited nodes)
        
    """

    def __init__(self, id: int, system_time: float, current_node: int, capacity: int): 
        self.id = id
        self.status = VehicleStatus.IDLE
        self.veh_time = system_time
        self.arc_completion_percentage = 0.0 #update when call move_to_time
        self.time_to_complete_current_arc = 0.0 #update when call move_to_time
        self.current_node = current_node 
        self.target_node = None
        self.route = []
        self.capacity = capacity
        self.load = 0
        self.schedule = []
        self.served_req_IDs = []
        self.run_time = 0.0
        self.assigned_reqs = [] #for debug
        self.visited_nodes = [] #for debug
        self.assigned_schedules = [] #for debug
        # self.time_marks_for_visited_nodes = []

    # schedule for trips, routes for each nodes need to be visited. 
    def move_to_time(self, current_system_time:float): 
        if len(self.schedule) == 0: #no more nodes to visit
            self.status = VehicleStatus.IDLE
            self.veh_time = current_system_time
        else:
            # update statistic
            self.run_time += (current_system_time - self.veh_time)
            while self.veh_time < current_system_time: #update veh_time until it reaches current_system_time
                
                assert self.current_node != None
                self.target_node = self.route[0]

                arc_time = get_timeCost(self.current_node, self.target_node)
                self.time_to_complete_current_arc = arc_time * (1 - self.arc_completion_percentage)

                if self.veh_time + self.time_to_complete_current_arc <= current_system_time: #able to complete the arc
                    self.veh_time += self.time_to_complete_current_arc
                    self.arc_completion_percentage = 0
                    self.current_node = self.target_node
                    self.visited_nodes.append(self.current_node)    

                    self.route.pop(0) #remove the visited node from route

                    if len(self.route) == 0: #no more nodes to visit, update schedule. Current node is an end node(node in schedule)
                        if self.schedule[0][1] == 1: #PU node
                            # [req.Ori_id, PU/DO, req.Num_people, req.Latest_PU/DO_Time, req.Shortest_TT, req.Req_ID, 'NEW_PU'/'NEW_DO']
                            self.load += self.schedule[0][2]
                        elif self.schedule[0][1] == -1: #DO node
                            self.load -= self.schedule[0][2]
                            if self.check_req_delivery(self.schedule[0]): #check if the request is delivered on time
                                self.served_req_IDs.append(self.schedule[0][5])
                                assert self.schedule[0][5] in self.assigned_reqs #DEBUG CODE, request should be in assigned_reqs
                            else: #request is not delivered on time
                                raise KeyError("Request is not delivered on time")
                        else: #Balancing node
                            self.status = VehicleStatus.IDLE # finish rebalancing, vehicle is idle
                            self.veh_time = current_system_time
                            self.schedule = []
                            self.route = []
                            break
                        
                        # [req_ID, node type, time of visiting]
                        # rebalance: req_ID and node type are 0       
                        # self.time_marks_for_visited_nodes.append([self.schedule[0][5], self.schedule[0][1], self.veh_time])
                        self.schedule.pop(0) #remove the visited node from schedule

                        if len(self.schedule) == 0: #no more nodes to visit
                            self.status = VehicleStatus.IDLE  
                            self.veh_time = current_system_time #vehicle will stay idle until next round of requests
                            assert set(self.assigned_reqs) == set(self.served_req_IDs) #DEBUG CODE, all assigned requests should be served
                            break
                        else: #update route
                            target_end_node = self.schedule[0][0]
                            self.update_route(target_end_node)
                            assert self.route != [] #DEBUG CODE, vehicle should have a route

                else: #unable to complete the arc
                    self.arc_completion_percentage += (current_system_time - self.veh_time)/arc_time 
                    self.veh_time = current_system_time


    def update_schedule(self, new_schedule: list):
        # assert len(new_schedule) < 5 #DEBUG CODE
        # assert self.route == []
        assert new_schedule != [] #DEBUG CODE, new_schedule should not be empty
        self.schedule = pickle.loads(pickle.dumps(new_schedule)) 
        self.assigned_schedules.append(new_schedule)
        # self.status = VehicleStatus.WORKING
        
        #update route according to the new schedule
        if len(self.route) == 0: # vehicle has no route
            if self.current_node == self.schedule[0][0]: #vehicle is at target end node
                self.route = [self.current_node]
                assert len(self.route) != 0 #DEBUG CODE, vehicle should have a route
            else:
                new_route = get_route(self.current_node, self.schedule[0][0])
                self.route = new_route
                assert len(self.route) != 0 #DEBUG CODE, vehicle should have a route
                
        else: # vehicle has a route, but overwrites it, self.target_node = self.route[0](where the vehicle is going to visit next)
            assert self.target_node != None 
            # self.target_node = self.route[0]
            new_route = get_route(self.target_node, self.schedule[0][0])
            self.route = new_route  
            assert len(self.route) != 0 #DEBUG CODE, vehicle should have a route


    def update_route(self, target_end_node):
        assert len(self.route) == 0 #DEBUG CODE, when trigger this, vehicle should have no route
    
        route_to_target = get_route(self.current_node, target_end_node)
        self.route = pickle.loads(pickle.dumps(route_to_target)) 
        
    def check_req_delivery(self, schedule_node):
        current_time = self.veh_time
        
        req_Latest_DO_Time = schedule_node[3]
        if current_time <= req_Latest_DO_Time:
            return True
        else:
            return False

    def remove_duplicate_sublists(self, lst):
        seen = set()
        result = []
        for sublist in lst:
            # Convert the sublist to a tuple to make it hashable
            sublist_tuple = tuple(sublist)
            # Check if the sublist is not in the set of seen sublists
            if sublist_tuple not in seen:
                # Add the sublist to the result list
                result.append(sublist)
                # Add the sublist to the set of seen sublists
                seen.add(sublist_tuple)
        return result
