# drone_proj
Skyviper Autonomous Drone Project code Repository 


Drone Project Code Python Script and Other Files
Raspberry Pi Zero Dependencies:

Librealsense - use this link:
https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md
If error occurs when following these instructions regarding illegal instruction, follow steps in this link:
https://github.com/IntelRealSense/librealsense/issues/3490


Drone script taken directly from https://github.com/thien94/vision_to_mavros/blob/master/scripts/t265_to_mavlink.py
and adapted to use dronekit and without some extra features, make sure that code is understood before this one, as it is more complete. 

When Debug Enabled in the script, messages are printed to console, and t265 data is logged to text file. The other python file t265graph.py, where the filename of the text file in the first line of code can be changed to graph the t265 xyz data.

