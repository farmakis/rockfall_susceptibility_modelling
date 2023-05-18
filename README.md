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
The training data for the models should represent rock slope areas that include both stable and unstable points (points that belong to rockfalls) resulted from point cloud based change detection after de-noising and clustering. Details on the data generation workflow are provided [here](https://www.mdpi.com/2220-9964/10/3/157). When both <code>rockfall</code>  and <code>non_rockfall</code> points are are detected, make sure that the extracted point clouds <code>(.txt)</code> containing the rockfall points, inlude the cluster ID in the last column. Also, the original rock slope models be  saved in <code>.pcd</code> format. Then copy them in the respective folders following the steps below:
  1) Create a folder called <code>data</code>
  2) In <code>data</code>, create two folders named <code>models</code> and <code>rockfalls</code>
  3) Paste the <code>.pcd</code> files of the raw point clouds in the <code>models</code> folder created in (2)
  4) Paste the <code>.txt</code> files of the rockfall points and the cluster IDs in the <code>rockfalls</code> folder created in (2).

  # <sub><sub>Naming convention
  For every change detection analysis, for instance, between January 1st, 2022 and January 1st, 2023, both the raw point clouds and the resulted rockfall file should be named after the dates, as follows:
  - Reference point cloud: 2022-01-01.pcd
  - Compared point cloud: 2023-01-01.pcd
  - Rockfall file: 2022-01-01_to_2023-01-01.txt
  
 Now, you are ready to sample the training examples and the Tensorflow records by executing:
 <pre><code>python parser.py
 </code></pre>
 <pre><code>python create_dataset.py --box_size ## --points_per_box ### --batch_size ####
 </code></pre>
 where <code>##</code> is the size of each sampling box in meters (default=10), <code>###</code> the number of points to be sampled from each box (default=512), and <code>####</code> the batch size of the dataset (default=16).
