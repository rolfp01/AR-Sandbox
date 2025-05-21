import numpy as np
import cv2
from openni import openni2

def read_Depth_Camera(color_stream, depth_stream):
    # Lese ein Frame vom Farbsensor
    color_frame = color_stream.read_frame()
    color_data = color_frame.get_buffer_as_uint8()
    color_image = np.frombuffer(color_data, dtype=np.uint8).reshape((color_frame.height, color_frame.width, 3))

    # Lese ein Frame vom Tiefensensor
    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_frame.height, depth_frame.width))

    return {"color" :color_image, "depth":depth_image}

def read_Depth_Camera_only_color(color_stream):
    # Lese ein Frame vom Farbsensor
    color_frame = color_stream.read_frame()
    color_data = color_frame.get_buffer_as_uint8()
    color_image = np.frombuffer(color_data, dtype=np.uint8).reshape((color_frame.height, color_frame.width, 3))
    return color_image
      
def read_Depth_Camera_only_depth(depth_stream):
    # Lese ein Frame vom Tiefensensor
    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_frame.height, depth_frame.width))
    return depth_image
    