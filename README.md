# Depth Map Prediction with Multi-Scale Deepnetwork

In the project, we propose to replicate the depth map prediction from a single image using the multi-scale deep network introduced by Eigen et al. [1] (2014) with pytorch implementation and compare our work with existing literature.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## File Explanation

### main.py
The main function of the project.

### model.py
The CNN model with coarse net and fine net.

### IO.py
Handles the input data.



### Prerequisites

What things you need to install in your python environment

```
import pytorch
```
```
import numpy as np
```
```
example
```

## Dataset
We use [NYU Depth V2]('https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html') provided by Nathan Siberman and Rob Fergus. The NYU-Depth data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect. The dataset contains 464 different indoor scenes with 26 scene types. There are 1446 densely labeled pairs of aligned RGB and depth frames. In addition to the labeled images, the dataset also contains a large number of new unlabeled images.


## Built With

* Python3 - The programming language used
* Jupyter notebook - The coding environment used


## Authors

* **Yuqi Zhang** 
* **Zheng Xu**
* **Rui Tang**
* **Xinyang Wang**
* **Sai Wu**

## Reference
* Eigen, D., Puhrsch, C., and Fergus, R. (2014). Depth map prediction from a single image using a multi-scale deep network. In Advances in neural information processing systems (pp. 2366-2374).
* NYU Depth V2 dataset, Available at: [https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html]
* K. Karsch, C. Liu, S. B. Kang, and N. England. Depth extraction from video using nonparametric
sampling. In TPAMI, 2014.
* L. Ladicky, J. Shi, and M. Pollefeys. Pulling things out of perspective. In CVPR, 2014.
* Eigen, D., and Fergus, R. (2015). Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2650-2658).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
