# Behaviorial Cloning Project

<img align="right" src="https://www.pyimagesearch.com/wp-content/uploads/2017/12/not_santa_detector_dl_logos.jpg" width="250px" />
<img src="https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg"/>

- **Teach your car how to drive by cloning your driving behavior**
- `Keras` + `Tensorflow` + `OpenCV` + [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) + `Unity` + `AWS GPU EC2`

---

<div align="center"><b>Autonomous Driving</b>&emsp;|&emsp;<a href="https://github.com/x65han/Behavioral-Cloning/blob/master/run1/output_video.mp4?raw=true">Full Video</a>&emsp;|&emsp;<a href="https://github.com/x65han/Behavioral-Cloning/blob/master/report.md">Full Report</a></div><br>
<div align="center"><img width="60%" src="https://github.com/x65han/Behavioral-Cloning/blob/master/run1/output_video.gif?raw=true"/></div><br>

## Overview

- The following image is a [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) used by **Nvidia** to build behavior cloning neural networks
- This image provides the basic structure for `behaviorial cloning` neural network used in this project with some tweaks such as the last `fully-connected layer`
- For more `installation` details, please refer to [installation.md](https://github.com/x65han/Behavioral-Cloning/blob/master/installation.md) file for more information
<div align="center"><img src="https://github.com/x65han/Behavioral-Cloning/blob/master/miscellaneous/conv_net_model.png?raw=true" width="60%" /></div>

<hr>
<br>

<div align="center"><b>Sample Input Images with Steering Angle</b></div>
<div align="center"><img src="https://github.com/x65han/Behavioral-Cloning/blob/master/miscellaneous/sample_input.png?raw=true" width="100%" /></div>

I used the following technique to **pre-process** input data for **faster** and more **accurate** learning
- Crop out car hood, sky, and trees (irrelevant to driving)
- Normalize images
- Apply steering correction of 0.2 on left and right cameras
- Flip images with curvature > 0.33 to augment data set

<div align="center"><b>Pre processed Input Images with Steering Angle</b></div>
<div align="center"><img src="https://github.com/x65han/Behavioral-Cloning/blob/master/miscellaneous/pre_processed.png?raw=true" width="100%" /></div>
