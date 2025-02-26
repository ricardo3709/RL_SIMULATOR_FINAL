
class Statistic(object):
    """
    Statistic is a class for statistics
    Attributes:
        total_requests: total number of requests
        total_time: total time
        total_wait_time: total waiting time
        total_detour: total detour
        total_served_requests: total number of served requests
        total_rejected_requests: total number of rejected requests
    """
    def __init__(self):
        self.total_requests = 0
        self.total_time = 0.0
        self.total_wait_time = 0.0
        self.total_detour = 0.0
        self.total_picked_requests = 0
        self.total_rejected_requests = 0
        self.total_pending_requests = 0
        self.total_veh_run_time = 0.0
        self.all_veh_position_series = []
        self.served_requests_IDs = []