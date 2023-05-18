# Rockfall Susceptibility Modelling
This is a 3D slope-scale rockfall susceptibility modelling (RSM) project using computer vision. RSM is approached as a point cloud semantic segmentation (PCSS) problem. 3D geometric learning neural networks are employed for analyzing points clouds to interpret high-resolution digital observations capturing the evolution of a rock slope via long-term, LiDAR-based differencing. The implementation includes the [PointNet++](https://arxiv.org/abs/1612.00593), [PointCNN](https://arxiv.org/abs/1801.07791), and [DGCNN](https://arxiv.org/abs/1801.07829) modules. Detailed applications of the models and analysis results on real rockfall monitoring cases are demonstrated in the associated research [paper].
The repository includes components of the TensorFlow 2 layers provided [here](https://github.com/dgriffiths3/pointnet2-tensorflow2) and the TensorFlow operations provided [here](https://github.com/charlesq34/pointnet2/tree/master/tf_ops).

# <sub>Installation
The implementations in the associated [paper](https://www.sciencedirect.com/science/article/pii/S0013795222003210) were done in a Ubuntu 18.04 OS with the following setup:
  - python 3.6
  - tensorflow-gpu 2.2.0
  - cuda 10.1
  
To compile the TensorFlow operations make sure the <code>CUDA_ROOT</code> path in <code>tf_ops/compile_ops.sh</code> points to the correct CUDA installation folder in your machine. Then compile the operations by executing the following commands in the project's directory:

<pre><code>chmod u+x tf_ops/compile_ops.sh
tf_ops/compile_ops.sh
</code></pre>

# <sub>Data preparation
The training data for the models should represent rock slope areas that include both stable and unstable points (points that belong to rockfalls) resulted from point cloud based change detection after de-noising and clustering. Details on the data generation workflow are provided [here](https://www.mdpi.com/2220-9964/10/3/157). When both <code>rockfall</code>  and <code>non_rockfall</code> points are are detected, make sure that the extracted point clouds <code>(.txt)</code> containing the rockfall points, inlude the cluster ID at the last column. Also, the original rock slope models be  saved in <code>.pcd</code> format.

  # Naming convention: should include the dafollowing the steps below:
  1) Create a folder called <code>data</code>
  2) In <code>data</code>, create two folders named <code>models</code> and <code>rockfall</code>
  3) In each of the folders created in (2), create 3 folders named <code>train</code>, <code>dev</code>, and <code>test</code> and copy the <code>.off</code>       file there in
