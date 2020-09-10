# Ubuntu 18.04 Environment Configuration (from beginning to giving up)
## 1. Install Ubuntu
* make sure that the /swap partition is twice larger than the computer memory
## 2. Follow ROS installation instruction
* http://wiki.ros.org/melodic/Installation/Ubuntu
## 3. Install Nvidia GPU driver
* (nvidia driver 430 for 1080 Ti) Follow the instruction: http://ubuntuhandbook.org/index.php/2019/04/nvidia-430-09-gtx-1650-support/
## 4. Install CUDA
* From CUDA official website to download CUDA-*-10.run file and install
* Add those command to your ~/.bashrc and comment other CUDA command:
```
export CUDA_HOME=/usr/local/cuda # change this to your cuda root folder

export CUDA_INC_PATH=${CUDA_HOME}/include
export CUDA_LIB_PATH=${CUDA_HOME}/lib64

export CUDA_INSTALL_PATH=${CUDA_HOME}

export PATH=${CUDA_HOME}/bin:$PATH
export PATH=${CUDA_HOME}/computeprof/bin:$PATH

export LD_LIBRARY_PATH=${CUDA_HOME}/computeprof/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export MANPATH=${CUDA_HOME}/man:$MANPATH

export OPENCL_HOME=${CUDA_HOME}
export OPENCL_INC_PATH=${OPENCL_HOME}/include
export OPENCL_LIB_PATH=${OPENCL_HOME}/lib64
export LD_LIBRARY_PATH=${OPENCL_LIB_PATH}:$LD_LIBRARY_PATH
```
## Install Anaconda3
* After installation, do not add anything to `~/.bashrc` because **it will influence ROS**!
* Add the following command to your `~/.bashrc`
```
alias condaenv=”export PATH=”/home/safeai/anaconda3/bin:$PATH””
```
When you want to use conda environment in the terminal, you only need to execute `condaenv` command for the first time, and then in this terminal, all the python path will be linked to your anconda environment.
* You may want to setup a python 3.6 conda environment which can be used in ROS, using following command:
```
conda create -n py36 python=3.6
```
Then add `alias py36=”source activate py36"` to your `~/.bashrc` file.
When you want to use py36 environment in the terminal, you only need to execute `condaenv` command for the first time, and then run `py36` in this terminal, now you can use this python environment!
## How to use ROS in anaconda py36 environment
* Reference: https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674
* First of all, make sure you are in the `py36` environment
* Install the following missing libraries:
```
sudo apt-get install python3-pip python3-yaml
pip install rospkg catkin_pkg
```
* Now you can test in your terminal:
```
python
import rospy
```
If no error message, then it is OK.
### * Use cv_bridge in py36 environment
* If you directly use cv_bridge in python3, you may have following issue:
```
File “/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py”, line 91, in encoding_to_cvtype2
from cv_bridge.boost.cv_bridge_boost import getCvType
ImportError: dynamic module does not define module export function (PyInit_cv_bridge_boost)
```
This is because the cv_bridge module is compiled via python2.7.
* To solve this, let’s install opencv in anaconda first and some tools as well.
```
conda install -c conda-forge opencv
sudo apt-get install python-catkin-tools python3-dev python3-numpy
```
* Then create a new workspace for ros packages which you want to compile by python3.6
```
cd ~/ && mkdir py3_ws && cd py3_ws
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
mkdir src
catkin config — install
cd src
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
cd ..
catkin build cv_bridge
```
Now you can use cv_bridge in python3! (But you need to source this workspace first)
* How to verify?
create a `test.py` file and paste the following code into it:
```
#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
class image_converter:
def __init__(self):
self.image_pub = rospy.Publisher(“/image_topic2”,Image)
self.bridge = CvBridge()
self.image_sub = rospy.Subscriber(“/wide_stereo/left/image_raw_throttle”,Image,self.callback)
def callback(self,data):
try:
cv_image = self.bridge.imgmsg_to_cv2(data, “bgr8”)
except CvBridgeError as e:
print(e)
(rows,cols,channels) = cv_image.shape
if cols > 60 and rows > 60 :
cv2.circle(cv_image, (150,50), 20, 255)
cv2.imshow(“Image window”, cv_image)
cv2.waitKey(3)
try:
self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, “bgr8”))
except CvBridgeError as e:
print(e)
def main(args):
ic = image_converter()
rospy.init_node(‘image_converter’, anonymous=True)
try:
rospy.spin()
except KeyboardInterrupt:
print(“Shutting down”)
cv2.destroyAllWindows()
if __name__ == ‘__main__’:
main(sys.argv)
```
Then download any test dataset from https://projects.csail.mit.edu/stata/downloads.php , for example I downloaded the `2011–04–06–07–04–17.bag` rosbag, so I just start a `roscore` in a terminal, go to the rosbag folder and run
```
rosbag play 2011–04–06–07–04–17.bag
```
Then start a new terminal:
```
condaenv
py36
source ~/py3_ws/devel/setup.bash
```
Then execute the `test.py` file by `python test.py`
Hopefully you can see the result !
ROS can communicate with python3 node via cv_bridge! You can use any deep learning model now!
* For any ros packages you want to use in python3, you can try such kind of method.
Be sure that don’t create too much ros workspace because they may influence each other. My suggestion is **do not create more than 3 ros workspace in one computer!**
## Use Pytorch and cudnn in Anaconda
* Just `conda install pytorch`
## Tricks
Make your terminal `tab` completion not sensitive to the case and the command search more efficient:
```
sudo vim /etc/inputrc
```
add
```
set completion-ignore-case On
“\ep”: history-search-backward
“\e[A”: history-search-backward
“\e[B”: history-search-forward
```
