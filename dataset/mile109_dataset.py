import os
import open3d as o3d
import numpy as np
import random
import glob
import tensorflow as tf
import util.provider as provider
from util.point_cloud_util import load_labels

#Random split
# file_prefixes = [
#     "2013-11-28",
#     "2014-06-04",
#     "2014-09-03",
#     "2014-09-13",
#     "2014-11-10",
#     "2015-02-22",
#     "2015-04-03",
#     "2015-08-23",
#     "2015-10-22",
#     "2016-02-16",
#     "2016-07-25",
#     "2017-04-08",
#     "2017-09-04",
#     "2018-04-21",
#     "2018-06-27",
#     "2014-11-04",
#     "2015-06-11",
#     "2016-10-12",
#     "2018-05-31",
#     "2015-03-28",
#     "2016-05-07",
#     "2017-05-23",
#     "2018-08-02"
# ]
#
# split = random.sample(range(len(file_prefixes)), int(np.round(len(file_prefixes)*0.6)))
#
# train_file_prefixes = [file_prefixes[i] for i in split]
# dev_file_prefixes = [file_prefixes[i] for i in range(len(file_prefixes)) if not i in split]
# test_file_prefixes = ["2018-09-24"]


train_file_prefixes = [
    "2013-11-28",
    "2014-06-04",
    "2014-11-04",
    "2014-11-10",
    "2015-02-22",
    "2015-03-28",
    "2015-04-03",
    "2015-06-11",
    "2015-10-22",
    "2016-02-16",
    "2016-05-07",
    "2016-10-12",
    "2017-04-08",
    "2017-05-23",
    "2018-04-21",
    "2018-05-31",
]

dev_file_prefixes = [
    "2014-09-03",
    "2014-09-13",
    "2015-08-23",
    "2016-07-25",
    # "2017-09-04",
    "2018-06-27",
]

test_file_prefixes = [
    "2018-08-02",
    "2018-09-24",
]

all_file_prefixes = train_file_prefixes + dev_file_prefixes + test_file_prefixes

map_name_to_file_prefixes = {
    "train": train_file_prefixes,
    "train_full": train_file_prefixes + dev_file_prefixes,
    "dev": dev_file_prefixes,
    "test": test_file_prefixes,
    "all": all_file_prefixes,
}


def get_test_files():
    return test_file_prefixes


class Mile109FileData:
    def __init__(
            self, file_path_without_ext, has_labels, use_normals, box_size_x, box_size_y, num_classes
    ):
        """
        Loads file data
        """
        self.file_path_without_ext = file_path_without_ext
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y

        # Load points
        pcd = o3d.io.read_point_cloud(file_path_without_ext + ".pcd")
        self.points = np.asarray(pcd.points)

        # Load label. In pure test set, fill with zeros.
        if has_labels:
            self.labels = load_labels(file_path_without_ext + ".labels", num_classes)
        else:
            self.labels = np.zeros(len(self.points)).astype(bool)

        # Load normals. If not use_normals, fill with zeros.
        if use_normals:
            self.normals = np.asarray(pcd.normals)
        else:
            self.normals = np.zeros_like(self.normals)

        # Sort according to x to speed up computation of boxes and z-boxes
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.normals = self.normals[sort_idx]

    def _get_fix_sized_sample_mask(self, points, num_points_per_sample):
        """
        Get down-sample or up-sample mask to sample points to num_points_per_sample
        """
        # TODO: change this to numpy's build-in functions
        # Shuffling or up-sampling if needed
        if len(points) - num_points_per_sample > 0:
            true_array = np.ones(num_points_per_sample, dtype=bool)
            false_array = np.zeros(len(points) - num_points_per_sample, dtype=bool)
            sample_mask = np.concatenate((true_array, false_array), axis=0)
            np.random.shuffle(sample_mask)
        else:
            # Not enough points, recopy the data until there are enough points
            sample_mask = np.arange(len(points))
            while len(sample_mask) < num_points_per_sample:
                sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
            sample_mask = sample_mask[:num_points_per_sample]
        return sample_mask

    def _center_box(self, points):
        # Shift the box so that z = 0 is the min and x = 0 and y = 0 is the box center
        # E.g. if box_size_x == box_size_y == 10, then the new mins are (-5, -5, 0)
        box_min = np.min(points, axis=0)
        shift = np.array(
            [
                box_min[0] + self.box_size_x / 2,
                box_min[1] + self.box_size_y / 2,
                box_min[2],
            ]
        )
        points_centered = points - shift
        return points_centered

    def _extract_z_box(self, center_point):
        """
        Crop along z axis (vertical) from the center_point.

        Args:
            center_point: only x and y coordinates will be used
            points: points (n * 3)
            scene_idx: scene index to get the min and max of the whole scene
        """
        # TODO TAKES LOT OF TIME !! THINK OF AN ALTERNATIVE !
        scene_z_size = np.max(self.points, axis=0)[2] - np.min(self.points, axis=0)[2]
        box_min = center_point - [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]
        box_max = center_point + [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]

        i_min = np.searchsorted(self.points[:, 0], box_min[0])
        i_max = np.searchsorted(self.points[:, 0], box_max[0])
        mask = (
            np.sum(
                (self.points[i_min:i_max, :] >= box_min)
                * (self.points[i_min:i_max, :] <= box_max),
                axis=1,
            )
            == 3
        )
        mask = np.hstack(
            (
                np.zeros(i_min, dtype=bool),
                mask,
                np.zeros(len(self.points) - i_max, dtype=bool),
            )
        )

        # mask = np.sum((points>=box_min)*(points<=box_max),axis=1) == 3
        assert np.sum(mask) != 0
        return mask

    def sample(self, num_points_per_sample):
        points = self.points

        # Pick a point, and crop a z-box around
        center_point = points[np.random.randint(0, len(points))]
        scene_extract_mask = self._extract_z_box(center_point)
        points = points[scene_extract_mask]
        labels = self.labels[scene_extract_mask]
        normals = self.normals[scene_extract_mask]

        sample_mask = self._get_fix_sized_sample_mask(points, num_points_per_sample)
        points = points[sample_mask]
        labels = labels[sample_mask]
        normals = normals[sample_mask]

        # Shift the points, such that min(z) == 0, and x = 0 and y = 0 is the center
        # This canonical column is used for both training and inference
        points_centered = self._center_box(points)

        return points_centered, points, labels, normals

    def sample_batch(self, batch_size, num_points_per_sample):
        """
        TODO: change this to stack instead of extend
        """
        batch_points_centered = []
        batch_points_raw = []
        batch_labels = []
        batch_normals = []

        for _ in range(batch_size):
            points_centered, points_raw, gt_labels, normals = self.sample(
                num_points_per_sample
            )
            batch_points_centered.append(points_centered)
            batch_points_raw.append(points_raw)
            batch_labels.append(gt_labels)
            batch_normals.append(normals)

        return (
            np.array(batch_points_centered),
            np.array(batch_points_raw),
            np.array(batch_labels),
            np.array(batch_normals),
        )


class Mile109Dataset:
    def __init__(self, num_points_per_sample, split, use_normals, has_labels, box_size_x, box_size_y, path):
        """Create a dataset holder
        num_points_per_sample (int): Defaults to 8192. The number of point per sample
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size_x (int): Defaults to 10. The size of the extracted cube.
        box_size_y (int): Defaults to 10. The size of the extracted cube.
        path (float): Defaults to 'dataset/semantic_data/'.
        """
        # Dataset parameters
        self.num_points_per_sample = num_points_per_sample
        self.split = split
        self.use_normals = use_normals
        self.has_labels = has_labels
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.num_classes = 2
        self.path = path
        self.labels_names = [
            "non-rockfall",
            "rockfall"
        ]

        # Get file_prefixes
        file_prefixes = map_name_to_file_prefixes[self.split]
        print("Dataset split:", self.split)
        print("Loading file_prefixes:", file_prefixes)

        # Load files
        self.list_file_data = []
        for file_prefix in file_prefixes:
            if not has_labels:
                file_prefix = file_prefix + "_full"
            file_path_without_ext = os.path.join(self.path, file_prefix)
            file_data = Mile109FileData(
                file_path_without_ext=file_path_without_ext,
                use_normals=self.use_normals,
                has_labels=self.has_labels,
                box_size_x=self.box_size_x,
                box_size_y=self.box_size_y,
                num_classes=self.num_classes
            )
            self.list_file_data.append(file_data)

        # Pre-compute the probability of picking a scene
        self.num_scenes = len(self.list_file_data)
        self.scene_probas = [
            len(fd.points) / self.get_total_num_points() for fd in self.list_file_data
        ]

    def sample_batch_in_all_files(self, batch_size):
        batch_data = []
        batch_label = []

        for _ in range(batch_size):
            points_centered, labels, normals = self.sample_in_all_files()

            batch_data.append(np.hstack((points_centered, normals)))
            batch_label.append(labels)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)

        return batch_data, batch_label

    def sample_in_all_files(self):
        """
        Returns points and other info within a z - cropped box.
        """
        # Pick a scene, scenes with more points are more likely to be chosen
        scene_index = np.random.choice(
            np.arange(0, len(self.list_file_data)), p=self.scene_probas
        )
        # Sample from the selected scene
        points_centered, points_raw, labels, normals = self.list_file_data[
            scene_index
        ].sample(num_points_per_sample=self.num_points_per_sample)

        # return points_centered, labels
        return points_centered, labels, normals

    def get_total_num_points(self):
        list_num_points = [len(fd.points) for fd in self.list_file_data]
        return np.sum(list_num_points)

    def get_num_batches(self, batch_size):
        return int(
            self.get_total_num_points() / (batch_size * self.num_points_per_sample)
        )

    def get_file_paths_without_ext(self):
        return [file_data.file_path_without_ext for file_data in self.list_file_data]

