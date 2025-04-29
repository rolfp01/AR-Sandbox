import numpy as np

def read_Depth_Camera(pipeline):
    # Warten auf neue Frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Frames in numpy-Arrays umwandeln
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return {"color" :color_image, "depth":depth_image}
    