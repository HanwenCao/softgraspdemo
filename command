mkdir -p catkin_ws/src
cd catkin_ws/src/
git clone https://github.com/HanwenCao/softgraspdemo.git
git clone https://github.com/eric-wieser/ros_numpy.git
cd .. (now should be in catkin_ws/)
catkin build
source ./devel/setup.bash
cd ./src/yolov5_test/scripts/
rosrun yolov5_test detect_sub.py


******************YOLOv5**********************
# bringup camera
roslaunch realsense2_camera rs_camera.launch align_depth:=true

# yolo
source ~/test_ros_ws/devel/setup.bash
cd ~/test_ros_ws/src/yolov5_test/scripts

rosrun yolov5_test detect_sub.py 

(for testing) rosrun yolov5_test realsense_subscriber.py


******************point cloud**************
# bringup camera
roslaunch realsense2_camera rs_camera.launch filters:=pointcloud
rostopic info /camera/depth/color/points

***segmentation:***
rosrun obj_recognition obj_recognition_segmentation 

*** view pcd ***
pcl_viewer -multiview 1 <pcd_filepath>
