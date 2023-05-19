'''
This script takes the change detection campaigns and generated pre-failure state and binary rockfall labels
based on date correspondence
'''

import os
import numpy as np
import open3d as o3d
import argparse
from progress.bar import Bar

from util.point_cloud_util import write_labels, compute_normals


# Global arg collections
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='wcw')

args = parser.parse_args()

if args.dataset == "mile109":
    from dataset.mile109_dataset import get_test_files
elif args.dataset == "wcw":
    from dataset.wcw_dataset import get_test_files
elif args.dataset == "marsden":
    from dataset.marsden_dataset import get_test_files


def point_cloud_txt_to_pcd(txt):
    # txt: x y z _ _ _ _
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(txt[:, :3])
    return pcd


def sample_data(model, rockfalls, radius):
    """
    Input types
    :param model: pcd file
    :param rockfalls: txt file with the last column being the rockfall cluster id
    :param radius: the sampling area radius around each rockfall's center
    :return: pcd file
    """
    sampled = o3d.geometry.PointCloud()
    temp = o3d.geometry.PointCloud()
    model_tree = o3d.geometry.KDTreeFlann(model)
    for id in np.unique(rockfalls[:, -1]):
        points_id = np.where(rockfalls[:, -1] == id)
        if len(points_id[0]) > 2:
            points = rockfalls[points_id, :3][0]

            # # Sampling with scaled bb of the cluster
            temp.points = o3d.utility.Vector3dVector(points)
            bb = temp.get_axis_aligned_bounding_box()
            bb.scale(1.2, bb.get_center())
            points_per_box.append(len(model.crop(bb).points))
            sampled += model.crop(bb)

            # # Sampling with predefined-sized bb of the clusters
            # centroid = np.mean(points, axis=0)
            # bb = o3d.geometry.AxisAlignedBoundingBox(centroid-radius, centroid+radius)
            # points_per_box.append(len(model.crop(bb).points))
            # sampled += model.crop(bb)

            # # Sampling with predefined-sized sphere around each cluster
            # centroid = np.mean(points, axis=0)
            # [_, idx, _] = model_tree.search_radius_vector_3d(centroid, radius)
            # points_per_box.append(len(model.crop(bb).points))
            # sampled += model.select_by_index(idx)
        else:
            rockfalls = np.delete(rockfalls, points_id, axis=0)

    out_points = np.asarray(sampled.points)
    out_points = np.unique(out_points, axis=0)
    sampled.points = o3d.utility.Vector3dVector(out_points)
    if not sampled.has_normals():
        sampled = compute_normals(sampled)
    return sampled, rockfalls


def create_label_file(model, rockfalls):
    """
    Input types
    :param model: pcd file
    :param rockfalls: pcd file
    :return: labels file
    """
    model_tree = o3d.geometry.KDTreeFlann(model)
    labels = np.zeros(len(model.points))
    # labels = np.hstack((np.ones((len(model.points), 1)), np.zeros((len(model.points), 1))))
    for point in rockfalls.points:
        [_, idx, _] = model_tree.search_knn_vector_3d(point, 10)
        labels[np.asarray(idx)] = 1
        # labels[np.asarray(idx), 0] = 0
        # labels[np.asarray(idx), 1] = 1
    return labels.astype(float)


if __name__ == "__main__":

    dir = "/media/farmakis/Data/Rockfall_Susceptibility_Semantic_Segmentation/" + args.dataset
    model_dir = os.path.join(dir, "models")
    rockfall_dir = os.path.join(dir, "rockfalls")

    current_dir = os.getcwd()
    dataset_dir = os.path.join(current_dir, "dataset")
    raw_dir = os.path.join(dataset_dir, args.dataset)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    points_per_box = []

    # setup progress bar
    with Bar('Parsing Data', max=len(os.listdir(model_dir))) as bar:

        for model_txt, rockfall_txt in zip(os.listdir(model_dir), os.listdir(rockfall_dir)):
            file_prefix = model_txt.split(".")[0]

            if model_txt.split(".")[-1] == "pcd":
                model_pcd = o3d.io.read_point_cloud(os.path.join(model_dir, model_txt))
            else:
                try:
                    model_pcd = point_cloud_txt_to_pcd(np.loadtxt(os.path.join(model_dir, model_txt)))
                except ValueError:
                    model_pcd = point_cloud_txt_to_pcd(np.loadtxt(os.path.join(model_dir, model_txt), delimiter=","))

            try:
                rockfall_txt = np.loadtxt(os.path.join(rockfall_dir, rockfall_txt))
            except ValueError:
                rockfall_txt = np.loadtxt(os.path.join(rockfall_dir, rockfall_txt), delimiter=",")

            if file_prefix in get_test_files():
                model_pcd = compute_normals(model_pcd)
                o3d.io.write_point_cloud(os.path.join(raw_dir, file_prefix + "_full.pcd"), model_pcd)

            # Use this command to sample areas from the whole data to avoid non-rockfall bias
            model_pcd, rockfall_txt = sample_data(model_pcd, rockfall_txt, 5)
            rockfall_pcd = point_cloud_txt_to_pcd(rockfall_txt)
            labels = create_label_file(model_pcd, rockfall_pcd)

            write_labels(os.path.join(raw_dir, file_prefix + ".labels"), labels)
            o3d.io.write_point_cloud(os.path.join(raw_dir, file_prefix + ".pcd"), model_pcd)
            print(" --> {} data parsed".format(file_prefix))
            bar.next()

    print("num_points_per_box")
    print("mean: {}".format(np.mean(points_per_box)))
    print("min: {}".format(np.min(points_per_box)))
    print("max: {}".format(np.max(points_per_box)))


