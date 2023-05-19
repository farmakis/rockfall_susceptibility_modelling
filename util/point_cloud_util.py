import numpy as np
import open3d
import os


def _label_to_colors(labels):
    map_label_to_color = {
        0: [255, 255, 255],  # white
        1: [0, 0, 255],  # blue
        2: [128, 0, 0],  # maroon
        3: [255, 0, 255],  # fuchisia
        4: [0, 128, 0],  # green
        5: [255, 0, 0],  # red
        6: [128, 0, 128],  # purple
        7: [0, 0, 128],  # navy
        8: [128, 128, 0],  # olive
    }
    return np.array([map_label_to_color[label] for label in labels]).astype(np.int32)


def _label_to_colors_one_hot(labels):
    map_label_to_color = np.array(
        [
            [255, 255, 255],
            [0, 0, 255],
            [128, 0, 0],
            [255, 0, 255],
            [0, 128, 0],
            [255, 0, 0],
            [128, 0, 128],
            [0, 0, 128],
            [128, 128, 0],
        ]
    )
    num_labels = len(labels)
    labels_one_hot = np.zeros((num_labels, 9))
    labels_one_hot[np.arange(num_labels), labels] = 1
    return np.dot(labels_one_hot, map_label_to_color).astype(np.int32)


def colorize_point_cloud(point_cloud, labels):
    if len(point_cloud.points) != len(labels):
        raise ValueError("len(point_cloud.points) != len(labels)")
    if len(labels) < 1e6:
        print("_label_to_colors_one_hot used")
        colors = _label_to_colors_one_hot(labels)
    else:
        colors = _label_to_colors(labels)
    # np.testing.assert_equal(colors, colors_v2)
    point_cloud.colors = open3d.utility.Vector3dVector()  # Clear it to save memory
    point_cloud.colors = open3d.utility.Vector3dVector(colors)


def compute_normals(point_cloud):
    point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    point_cloud.orient_normals_to_align_with_direction()
    return point_cloud


# def load_labels(label_path):
#     # Assuming each line is a valid int
#     with open(label_path, "r") as f:
#         labels = [[float(line.split(" ")[0]), float(line.split(" ")[1])] for line in f]
#     return np.array(labels)

def load_labels(label_path, num_classes):
    # Assuming each line is a valid int
    with open(label_path, "r") as f:
        labels = [int(line) for line in f]
    one_hot = np.zeros((len(labels), num_classes))

    for point, label in enumerate(labels):
        one_hot[point, label] = 1

    return one_hot.astype(np.int32)


# def write_labels(label_path, labels):
#     with open(label_path, "w") as f:
#         f.write("\n".join(" ".join(map(str, x)) for x in labels))


def write_labels(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write("%d\n" % label)


def save_as_labeled_txt(in_dir, out_dir, file_prefix):
    labels = load_labels(os.path.join(in_dir, file_prefix + ".labels"))
    pcd = open3d.io.read_point_cloud(os.path.join(in_dir, file_prefix + ".pcd"))
    points = np.asarray(pcd.points)
    out_cloud = np.concatenate((points, labels.reshape(-1, 1)), axis=1)
    np.savetxt(os.path.join(out_dir, file_prefix + ".txt"), out_cloud, delimiter=",")
