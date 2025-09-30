import numpy as np
import pyrealsense2 as rs

def read_Depth_Camera(pipeline):
    # Warten auf neue Frames
    align_to = rs.stream.color
    align = rs.align(align_to)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()


    # Threshold Filter initialisieren
    threshold_filter = rs.threshold_filter()
    threshold_filter.set_option(rs.option.min_distance, 0.5)  # in Metern
    threshold_filter.set_option(rs.option.max_distance, 0.9)  # in Metern
    filtered_depth = threshold_filter.process(depth_frame)


    # Frames in numpy-Arrays umwandeln
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(filtered_depth.get_data())

    # Einfach: mit rechten Nachbarwerten füllen
    depth_image[:, :60] = depth_image[:, 60:120]

    return {"color" :color_image, "depth":depth_image}
    