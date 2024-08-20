from enum import IntEnum

def date_to_dirname(run_start_time):
    return str(run_start_time.date()) + "-" + str(run_start_time.hour) + "-" + str(run_start_time.minute) + "-" + str(run_start_time.minute)

class Action(IntEnum):
    SWIPE_UP = 0,
    SWIPE_DOWN = 1,
    SWIPE_LEFT = 2,
    SWIPE_RIGHT = 3
    DO_NOTHING = 4