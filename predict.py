import os
import sys
import argparse

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import open3d as o3d

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs Available: ", len(gpu_devices))
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
from util.tf_util import load_dataset, f1_score, precision, recall
tf.random.set_seed(1234)

# Global arg collections
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="pointnet++")
parser.add_argument("--logdir", type=str)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--evaluate", type=bool, default=False)
parser.add_argument("--save_eval", type=bool, default=False)
parser.add_argument("--visualize_samples", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--classes", type=int, default=2)
parser.add_argument("--box_size", type=int, default=10)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--points_per_box", type=int, default=512)

args = parser.parse_args()


def visualize_from_collector(pts, gt, pd, dir, idx):
    pts = pts[idx]
    gt = gt[idx][:,1].reshape(-1, 1)
    pd = np.where(pd[idx][:,1] > 0.5, 1, 0).reshape(-1, 1)
    out = np.concatenate((pts, gt, pd), axis=1)
    np.savetxt(os.path.join(dir, "out#{}.txt".format(idx)), out, delimiter=",", fmt='%1.8f')


def save_sample_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("sample.pcd", pcd)


def save_labeled_txt(filename, points, labels):
    out = np.concatenate((points, labels.reshape(-1, 1)), axis=1)
    np.savetxt("{}.txt".format(filename), out, delimiter=",", fmt='%.8f')


def predict(net):

    if args.logdir:
        logdir = args.model + "_" + args.logdir
    else:
        logdir = args.model

    if net == 'pointnet++':
        from models.pointnet import PointNet
        model = PointNet(batch_size=args.batch_size, num_classes=args.classes)
    elif net == 'pointcnn':
        from models.pointcnn import PointCNN
        from models.pointcnn import xconv_params, xdconv_params, fc_params
        model = PointCNN(batch_size=args.batch_size, num_classes=args.classes,
                         xconv_params=xconv_params, xdconv_params=xdconv_params, fc_params=fc_params)
    elif net == 'dgcnn':
        from models.dgcnn import DGCNN
        model = DGCNN(batch_size=args.batch_size, num_classes=args.classes,
                      num_points=args.points_per_box)

    model.build(input_shape=(args.batch_size, args.points_per_box, 6))
    print(model.summary())
    model.load_weights('./logs/{}/model/weights_{epoch:04d}.ckpt'.format(logdir, epoch=args.epoch)).expect_partial()

    if args.evaluate:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        dataset = load_dataset('dataset/test.tfrecord',
                               args.batch_size, args.points_per_box, args.classes)

        if args.save_eval: points_collector = []
        pd_labels_collector = []
        gt_labels_collector = []

        for points, labels in dataset.take(-1):
            if args.save_eval: points_collector.extend(points.numpy())
            pd_labels_collector.extend(model.predict(points, batch_size=args.batch_size))
            gt_labels_collector.extend(labels.numpy())

        if args.visualize_samples:
            output_dir = os.path.join("samples", logdir)
            os.makedirs(output_dir, exist_ok=True)
            for idx in range(len(points_collector)):
                visualize_from_collector(points_collector, gt_labels_collector, pd_labels_collector, output_dir, idx)

        gt = np.array(gt_labels_collector).reshape((-1, 2))[:, 1]
        pd = np.array(pd_labels_collector).reshape((-1, 2))[:, 1]

        m = tf.keras.metrics.MeanIoU(num_classes=args.classes)
        m.update_state(gt, np.where(pd > 0.5, 1, 0))

        fpr, tpr, thresholds = roc_curve(gt, pd)
        score = auc(fpr, tpr)
        print('AUC = {} / recall = {} / precision = {} / f1-score = {} / IoU = {}'.format(
            score, recall(gt, pd), precision(gt, pd), f1_score(gt, pd), m.result().numpy()))

        metrics = [score, recall(gt, pd), precision(gt, pd), f1_score(gt, pd), m.result().numpy()]

        if args.save_eval:
            output_dir = os.path.join("predictions", logdir, 'test')
            os.makedirs(output_dir, exist_ok=True)

            for id, (sample, gt, pd) in enumerate(zip(points_collector, gt_labels_collector, pd_labels_collector)):
                out = np.hstack((sample[:, :3], gt[:, 1].reshape(-1, 1), np.where(pd[:, 1] > 0.5, 1, 0).reshape(-1, 1)))
                np.savetxt(os.path.join(output_dir, 'sample_{}.txt'.format(id+1)), out, delimiter=",", fmt='%.8f')

            np.save('./predictions/roc_{}_{}.npy'.format(args.model, args.dataset), [fpr, tpr], allow_pickle=True)
            np.savetxt('./predictions/metrics_{}_{}_epoch_{}.txt'.format(args.model, args.dataset, args.epoch),
                       [metrics], header='AUC / recall / precision / f1-score / IoU',
                       delimiter=' / ', fmt='%.5f')

            plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label='{} / {} (AUC = {:.2f})'.format(args.model, args.dataset, score))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='lower right')
            plt.savefig('./predictions/roc_{}.png'.format(logdir))

    else:
        # Import dataset
        path = 'dataset/parsed/'
        from dataset.dataset import Dataset

        dataset = Dataset(
            num_points_per_sample=args.points_per_box,
            split='test',
            box_size_x=args.box_size,
            box_size_y=args.box_size,
            use_normals=1,
            has_labels=args.evaluate,
            path=path,
        )

        # Create output dirs
        output_dir = os.path.join("predictions", logdir)
        os.makedirs(output_dir, exist_ok=True)

        for file_data in dataset.list_file_data:
            print("Processing {}".format(file_data))
            points_collector = []
            pd_labels_collector = []

            # If num_samples < batch_size, will predict one batch
            for batch_index in range(int(np.ceil(args.num_samples / args.batch_size) - 1)):
                # Get data
                points_centered, points, gt_labels, normals = file_data.sample_batch(
                    batch_size=args.batch_size,
                    num_points_per_sample=args.points_per_box)

                input_data = np.concatenate((points_centered, normals), axis=-1)

                # Predict
                pd_labels = model.predict(input_data, batch_size=args.batch_size)

                # visualize_prediction(points, gt_labels, pd_labels)

                # Save to collector for file output
                points_collector.extend(points)
                pd_labels_collector.extend(pd_labels.reshape((-1, 2)))

            # Save point cloud and predicted labels
            file_prefix = os.path.basename(file_data.file_path_without_ext).split('_')[0]

            out_points = np.array(points_collector).reshape((-1, 3))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(out_points)
            pcd_path = os.path.join(output_dir, file_prefix + ".pcd")
            o3d.io.write_point_cloud(pcd_path, pcd)
            print("Exported pcd to {}".format(pcd_path))

            out_labels = np.asarray(pd_labels_collector)[:, 1]
            pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
            # np.savetxt(pd_labels_path, out_labels, fmt="%8d")
            np.savetxt(pd_labels_path, out_labels)
            print("Exported labels to {}".format(pd_labels_path))

            save_labeled_txt(os.path.join(output_dir, file_prefix), out_points, out_labels)


if __name__ == "__main__":
    predict(net=args.model)








