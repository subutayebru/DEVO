import h5py
import numpy as np
import csv
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from event_camera_py import Decoder 
from PIL import Image
import io
import argparse
import os
import math
import torch

from utils.event_utils import EventSlicer #compute_ms_to_idx
from collections import Counter

# Set up argument parser
parser = argparse.ArgumentParser(description="Process ROS bag file and extract data.")
parser.add_argument("bagpath", type=str, help="Path to the ROS bag file (.mcap)")
args = parser.parse_args()

# Use the provided path
bagpath = Path(args.bagpath)

# Check if the file exists
if not os.path.exists(bagpath):
    print(f"Error: File not found at {bagpath}")
    exit(1)

# Create output directory
output_dir = Path(bagpath)
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize the decoder
decoder = Decoder()

# Filenames for saving data
imu_file = output_dir / 'imu.csv'
image_timestamps_file = output_dir / 'images_timestamps.txt'
groundtruth_file = output_dir / 'stamped_groundtruth.txt'
#times_file = output_dir / 'times.txt'
images_dir = output_dir / 'images'
events_file = output_dir / 'events.h5'

# Ensure the image directory exists
Path(images_dir).mkdir(parents=True, exist_ok=True)

# Create a type store using the available ROS message types
typestore = get_typestore(Stores.ROS2_HUMBLE)


# List to store the parsed event data
all_events = []
image_count = 0
x_data=[]
y_data=[]
p_data=[]
t_data=[]

ev = np.dtype({
    'names': ['x', 'y', 'p', 't'],
    'formats': ['<u2', '<u2', 'i1', '<i4'],
    'offsets': [0, 2, 4, 8],
    'itemsize': 12
})


# Create reader instance and open for reading
with AnyReader([bagpath], default_typestore=typestore) as reader, \
     open(imu_file, 'w', newline='') as imu_csv, \
     open(image_timestamps_file, 'w') as img_ts_file, \
     open(groundtruth_file, 'w') as gt_file:
     #open(times_file, 'w') as time_file:
    
    # Set up CSV writer for IMU data
    imu_writer = csv.writer(imu_csv)
    imu_writer.writerow(['time', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 
                         'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z', 
                         'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])
    # Iterate over connections and process each topic
    for connection in reader.connections:
        print(f"Processing topic: {connection.topic} | Type: {connection.msgtype}")

        # Process event camera messages
        if connection.topic == '/uuv02/event_camera/events':
            for _, timestamp, rawdata in reader.messages(connections=[connection]):
                #print(f'time', timestamp)
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                if msg.__msgtype__ == 'event_camera_msgs/msg/EventPacket':
                    decoder.decode(msg)
                    events_cd = decoder.get_cd_events()
                    #print(f'Now appending for', len(events_cd))
                    #uzunluk = uzunluk+len(events_cd)

                    #print(f'Now appending for', len(events_cd), events_cd[0], events_cd[-1]) 
                    if events_cd is not None and len(events_cd) > 0:
                        #for row in events_cd:
                            #all_events.append(row)
                        all_events.append(events_cd)
                    #print(events_cd[3])    
                    #print(len(all_events))
                    #print(uzunluk)
            
        elif connection.topic == '/uuv02/vertical_camera/image_raw/compressed':
            for _, timestamp, rawdata in reader.messages(connections=[connection]):
                msg = reader.deserialize(rawdata, connection.msgtype)
                img_data = msg.data
                img = Image.open(io.BytesIO(img_data))
                img_filename = f'{images_dir}/frame_{image_count:010}.png'
                #img.save(img_filename)
                #print(f'ts for images',timestamp/1e3)
                img_ts_file.write(f'{timestamp/1e3}\n')
                image_count += 1
            print(f"Saved {image_count} images")
            
        # Process IMU data and write to CSV
        elif connection.topic == '/uuv02/event_camera/imu':
            for _, timestamp, rawdata in reader.messages(connections=[connection]):
                msg = reader.deserialize(rawdata, connection.msgtype)
                imu_writer.writerow([timestamp, 
                                     msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w, 
                                     msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, 
                                     msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        # Process ground truth odometry data and write to text file
        elif connection.topic == '/uuv02/ground_truth/odometry':
            for _, timestamp, rawdata in reader.messages(connections=[connection]):
                msg = reader.deserialize(rawdata, connection.msgtype)
                #print(f'ts for gt',timestamp/1e3)
                gt_file.write(f'{timestamp/1e3} {msg.pose.pose.position.x} {msg.pose.pose.position.y} {msg.pose.pose.position.z} '
                              f'{msg.pose.pose.orientation.x} {msg.pose.pose.orientation.y} {msg.pose.pose.orientation.z} {msg.pose.pose.orientation.w}\n')
                
    
    if all_events:
        x_vals = np.concatenate([events['x'] for events in all_events], dtype=np.int32)
        y_vals = np.concatenate([events['y'] for events in all_events], dtype=np.int32)
        p_vals = np.concatenate([events['p'] for events in all_events], dtype=np.int8)
        t_vals = np.concatenate([events['t'] + 1712260173484000  for events in all_events], dtype=np.int64)
    
    #1712259903863000  FOR circles_darker_0.30
    #1712260046902000  FOR circles_darker_0.20
    #1712260173484000  FOR circles_darker_0.10
    #1712259517589000  FOR circles_dark_0.20
    #1712259729142000  FOR circles_dark_0.30
    #1728646553198000  FOR manual
    #-----------------------------------------
    #1728658335807000  FOR Obj_data
    #1728658430157000  For Obj_angled
    #1728658540150000  For Obj_rotation 
    #1728657680820000  For Man_rot_robot
    #1728657960892000  For Man_rot_dustbin
    #1728656978404000  For Man_obj_5
    #1728656741601000  For Man_obj_4 
    #1728657353978000  For Man_obj_2
    #1728657208728000  For Man_obj_1
    #1728655888807000  For Man_4
    #1728655824962000  For Man_3
    #1728656022687000  For Man_2


    with h5py.File(events_file, 'w') as hf:
        #hf.create_dataset('ms_to_idx', data=ms_to_idx, dtype="<u8")
        hf.create_dataset('x', data=x_vals)
        hf.create_dataset('y', data=y_vals)
        hf.create_dataset('p', data=p_vals)
        hf.create_dataset('t', data=t_vals)
        #hf.create_dataset('ms_to_idx', data=ms_to_idx)

    print(f'Successfully saved events to events.h5 with keys: x, y, p, t')
    
    hf.close()
    print(f"Finished processing {bagpath}\n\n")
