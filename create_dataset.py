'''
This script generates TFRecords train/dev dataset from the parsed data
(see parser.py)
'''
import os
import sys
import multiprocessing as mp
import time
import argparse

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

# from dataset.semantic_dataset import SemanticDataset
tf.random.set_seed(1234)

# Global arg collections
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, default='wcw')
parser.add_argument("--box_size", type=int, default=10)
parser.add_argument("--points_per_box", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()

# Import dataset
path = 'dataset/' + args.dataset + "/"

if args.dataset == "mile109":
    from dataset.mile109_dataset import Mile109Dataset as Dataset
elif args.dataset == "wcw":
    from dataset.wcw_dataset import WCWDataset as Dataset
elif args.dataset == "marsden":
    from dataset.marsden_dataset import MarsdenDataset as Dataset

# Import dataset
TRAIN_DATASET = Dataset(
    num_points_per_sample=args.points_per_box,
    split='train',
    box_size_x=args.box_size,
    box_size_y=args.box_size,
    use_normals=1,
    has_labels=True,
    path=path,
)
DEV_DATASET = Dataset(
    num_points_per_sample=args.points_per_box,
    split='dev',
    box_size_x=args.box_size,
    box_size_y=args.box_size,
    use_normals=1,
    has_labels=True,
    path=path,
)
TEST_DATASET = Dataset(
    num_points_per_sample=args.points_per_box,
    split='test',
    box_size_x=args.box_size,
    box_size_y=args.box_size,
    use_normals=1,
    has_labels=True,
    path=path,
)
NUM_CLASSES = TRAIN_DATASET.num_classes


def get_batch(split):
    np.random.seed()
    if split == "train":
        return TRAIN_DATASET.sample_batch_in_all_files(args.batch_size)
    elif split == 'dev':
        return DEV_DATASET.sample_batch_in_all_files(args.batch_size)
    elif split == 'test':
        return TEST_DATASET.sample_batch_in_all_files(args.batch_size)


def fill_queues(stack_train, stack_dev, num_train_batches, num_dev_batches):
    """
    Args:
        stack_train: mp.Queue to be filled asynchronously
        stack_dev: mp.Queue to be filled asynchronously
        num_train_batches: total number of training batches
        num_dev_batches: total number of dev batches
    """
    pool = mp.Pool(processes=mp.cpu_count())
    launched_train = 0
    launched_dev = 0
    results_train = []  # Temp buffer before filling the stack_train
    results_dev = []  # Temp buffer before filling the stack_dev
    # Launch as much as n
    while True:
        if stack_train.qsize() + launched_train < num_train_batches:
            results_train.append(pool.apply_async(get_batch, args=("train",)))
            launched_train += 1
        elif stack_dev.qsize() + launched_dev < num_dev_batches:
            results_dev.append(pool.apply_async(get_batch, args=("dev",)))
            launched_dev += 1
        for p in results_train:
            if p.ready():
                stack_train.put(p.get())
                results_train.remove(p)
                launched_train -= 1
        for p in results_dev:
            if p.ready():
                stack_dev.put(p.get())
                results_dev.remove(p)
                launched_dev -= 1
        # Stability
        time.sleep(0.01)


def stack_data():
    """
    Returns:
        stacker: mp.Process object
        stack_validation: mp.Queue, use stack_validation.get() to read a batch
        stack_train: mp.Queue, use stack_train.get() to read a batch
    """
    with tf.device("/cpu:0"):
        # Queues that contain several batches in advance
        num_train_batches = TRAIN_DATASET.get_num_batches(args.batch_size)
        num_dev_batches = DEV_DATASET.get_num_batches(args.batch_size)
        stack_train = mp.Queue(num_train_batches)
        stack_dev = mp.Queue(num_dev_batches)
        stacker = mp.Process(target=fill_queues,
                             args=(stack_train, stack_dev, num_train_batches, num_dev_batches))
        stacker.start()
        return stacker, stack_train, stack_dev


def parser():

    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32 or dtype_ == np.int32 or dtype_ == np.bool:
            return tf.train.Feature(float_list=tf.train.FloatList(value=np.float32(ndarray).flatten().tolist()))

    def pcloud_example(points, labels):
        feature = {'points': _dtype_feature(points),
                   'labels': _dtype_feature(labels)}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    stacker, stack_train, stack_dev = stack_data()

    with tf.io.TFRecordWriter('dataset/'+args.dataset+'_train.tfrecord') as writer:
        print('::: Parsing train dataset')
        num_batches = TRAIN_DATASET.get_num_batches(args.batch_size)
        for batch_idx in range(num_batches):
            batch_data, batch_labels = stack_train.get()
            for f in range(args.batch_size):
                tf_example = pcloud_example(batch_data[f], batch_labels[f])
                writer.write(tf_example.SerializeToString())

    with tf.io.TFRecordWriter('dataset/'+args.dataset+'_dev.tfrecord') as writer:
        print('::: Parsing dev dataset')
        num_batches = DEV_DATASET.get_num_batches(args.batch_size)
        for batch_idx in range(num_batches):
            batch_data, batch_labels = stack_dev.get()
            for f in range(args.batch_size):
                tf_example = pcloud_example(batch_data[f], batch_labels[f])
                writer.write(tf_example.SerializeToString())

    with tf.io.TFRecordWriter('dataset/'+args.dataset+'_test.tfrecord') as writer:
        print('::: Parsing test dataset')
        num_batches = TEST_DATASET.get_num_batches(args.batch_size)
        for batch_idx in range(num_batches):
            batch_data, batch_labels = stack_dev.get()
            for f in range(args.batch_size):
                tf_example = pcloud_example(batch_data[f], batch_labels[f])
                writer.write(tf_example.SerializeToString())

    stacker.terminate()


if __name__ == "__main__":
    parser()
