from lane import Line
import logging

def init():
    global left_lane
    global right_lane
    left_lane = Line()
    right_lane = Line()

    # global my_logger
    # handler = logging.FileHandler('pipeline.log')
    # handler.setLevel(logging.INFO)
    # my_logger = logging.getLogger(__name__)
    # my_logger.addHandler(handler)
