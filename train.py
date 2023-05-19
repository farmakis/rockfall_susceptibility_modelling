import os
import sys
import argparse

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs Available: ", len(gpu_devices))
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

from util.tf_util import load_dataset, f1_score, precision, recall
tf.random.set_seed(1234)

# Global arg collections
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="dgcnn")
parser.add_argument("--dataset", type=str, default="mile109")
parser.add_argument("--classes", type=int, default=2)
parser.add_argument("--logdir", type=str)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--points_per_box", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=100)

args = parser.parse_args()


def train(net):

    if args.logdir:
        logdir = args.dataset + "_" + args.model + "_" + args.logdir
    else:
        logdir = args.dataset + "_" + args.model

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

    train_ds = load_dataset('dataset/'+args.dataset+'_train.tfrecord',
                            args.batch_size, args.points_per_box, args.classes)
    dev_ds = load_dataset('dataset/'+args.dataset+'_dev.tfrecord',
                          args.batch_size, args.points_per_box, args.classes)

    callbacks = [
        keras.callbacks.TensorBoard(
            './logs/{}'.format(logdir), update_freq='epoch', write_images=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join('./logs/{}/model'.format(logdir), 'weights_{epoch:04d}.ckpt'),
            monitor='val_loss', verbose=0, save_freq='epoch')
    ]

    model.build(input_shape=(args.batch_size, args.points_per_box, 6))
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["acc", f1_score, precision, recall]
    )

    model.fit(
        train_ds,
        validation_data=dev_ds,
        callbacks=callbacks,
        epochs=args.epochs,
        verbose=1
    )


if __name__ == "__main__":
    train(net=args.model)

