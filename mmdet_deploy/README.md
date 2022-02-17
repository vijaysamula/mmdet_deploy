# mmdet_deploy

This repository represents the real-time implementation of 2D object detection by use of [mmdetection](https://github.com/open-mmlab/mmdetection.git). The execution is carried out entirely in [docker](https://www.docker.com/).

## Docker 
```
https://github.com/vijaysamula/mmdet_deploy.git
docker build -t mmdet .
```

#### 1) To create container of the image.
```
nvidia-docker run  -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:$HOME/.Xauthority -v /path/to/shared_dir:/shared/ --net=host --pid=host --ipc=host --cap-add=SYS_PTRACE --name mmdet_cont mmdet:latest /bin/bash
```

#### 2) Open the container and follow the below procedure to run.
```
pip3 install -r requirements.txt
cd mmdet_deploy
catkin_make
source devel/setup.bash
```

#### 3) Model weights and config files can be downloaded from [mmdetection](https://github.com/open-mmlab/mmdetection.git) and [CBNetV2](https://github.com/VDIGPKU/CBNetV2.git).

#### 4) Finally run the launch file.
```
roslaunch mmdet_deploy mmdet_start_rgb.launch
```