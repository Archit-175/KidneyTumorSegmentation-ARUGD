# kits23_aru_gd_full_in_memory.py
# Adapted to load entire HDF5 datasets into RAM (no chunking) and to print GPU/CPU usage info.
# This version ALLOWS TensorFlow to allocate the GPU memory fully (no memory growth) and DOES NOT use mixed precision.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Conv2D, Activation, Input, Dropout, MaxPooling2D, UpSampling2D,
    Conv2DTranspose, Concatenate, multiply, Add, LeakyReLU, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MonteCarloDropout(Dropout):
    def call(self, inputs, training=None):
        if training is None:
            training = True  # Force dropout application for Monte Carlo during inference
        return super(MonteCarloDropout, self).call(inputs, training)

# -----------------------------------------
# GPU / System check
# -----------------------------------------
# ------------------------------------------------------------
# GPU memory limit to match shard allocation (10 shards ≈ 10GB)
# ------------------------------------------------------------
import tensorflow as tf

SHARD_COUNT = 20          # <-- IMPORTANT
TOTAL_SHARDS = 94         # Do not change
TOTAL_GPU_MEMORY_GB = 94  # H100 NVL memory size
ALLOCATED_GPU_MEMORY_GB = (SHARD_COUNT / TOTAL_SHARDS) * TOTAL_GPU_MEMORY_GB
memory_limit_mb = int(ALLOCATED_GPU_MEMORY_GB * 1024)

print(f"\n>>> Limiting TensorFlow to {ALLOCATED_GPU_MEMORY_GB:.1f} GB ({memory_limit_mb} MB) GPU memory\n")

physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        tf.config.set_logical_device_configuration(
            physical_gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
        )
        print("✅ GPU memory cap applied successfully.")
    except Exception as e:
        print("⚠ Could not set memory limit:", e)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TRAIN_HDF5_PATH = '/home/tanmoyhazra/h5_kidney_dataset_monai/train_data.h5'
VALID_HDF5_PATH = '/home/tanmoyhazra/h5_kidney_dataset_monai/valid_data.h5'
TEST_HDF5_PATH  = '/home/tanmoyhazra/h5_kidney_dataset_monai/test_data.h5'

IMG_HEIGHT = 256
IMG_WIDTH = 320
IMG_CHANNELS = 1
NUM_CLASSES = 4
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.2

LOSS_WEIGHTS_VEC = tf.constant([0.5, 1.0, 3.0, 2.0], dtype=tf.float32)

MODEL_SAVE_PATH = "/home/tanmoyhazra/h5_kidney_dataset_monai/saved_model/kits23_arunet_model_v2.h5"

# =============================================================================
# 2. LOAD ENTIRE HDF5 INTO RAM
# =============================================================================
print("--- Loading entire datasets into RAM (this will use a lot of memory) ---")
with h5py.File(TRAIN_HDF5_PATH, 'r') as f:
    X_train = f['X'][:]  # load all
    Y_train = f['Y'][:]
with h5py.File(VALID_HDF5_PATH, 'r') as f:
    X_valid = f['X'][:]
    Y_valid = f['Y'][:]
with h5py.File(TEST_HDF5_PATH, 'r') as f:
    X_test = f['X'][:]
    Y_test = f['Y'][:]

# Ensure correct dtypes
X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
X_test = X_test.astype(np.float32)

Y_train = Y_train.astype(np.float32)
Y_valid = Y_valid.astype(np.float32)
Y_test = Y_test.astype(np.float32)

num_train_samples = X_train.shape[0]
num_valid_samples = X_valid.shape[0]
num_test_samples = X_test.shape[0]

steps_per_epoch = num_train_samples // BATCH_SIZE
validation_steps = num_valid_samples // BATCH_SIZE
test_steps = num_test_samples // BATCH_SIZE

print(f"Training samples: {num_train_samples}, Steps per epoch: {steps_per_epoch}")
print(f"Validation samples: {num_valid_samples}, Validation steps: {validation_steps}")
print(f"Test samples: {num_test_samples}, Test steps: {test_steps}")

# --- Augmentation generators (using keras ImageDataGenerator but mapped via tf.py_function) ---
image_datagen = ImageDataGenerator(
    rotation_range=10, horizontal_flip=True, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.1, zoom_range=0.1
)
mask_datagen = ImageDataGenerator(
    rotation_range=10, horizontal_flip=True, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.1, zoom_range=0.1
)

# --- safer augmentation functions to replace your current ones ---

def augment_py_func(image, label):
    """
    Runs in Python (called from tf.py_function).
    Convert tensors -> numpy arrays, fix channel dims if needed,
    apply ImageDataGenerator transforms, return numpy float32 arrays.
    """
    # convert EagerTensor to numpy array (or ensure numpy array)
    if hasattr(image, "numpy"):
        image = image.numpy()
    else:
        image = np.asarray(image)

    if hasattr(label, "numpy"):
        label = label.numpy()
    else:
        label = np.asarray(label)

    # Ensure channel-last and proper dims
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if label.ndim == 2:
        label = label[..., np.newaxis]

    # Make contiguous arrays for numpy ops
    image = np.ascontiguousarray(image)
    label = np.ascontiguousarray(label)

    seed = np.random.randint(0, 10000)
    image = image_datagen.random_transform(image, seed=seed)
    label = mask_datagen.random_transform(label, seed=seed)

    return image.astype(np.float32), label.astype(np.float32)


def augment_data(image, label):
    """
    tf.data map function that calls the python augmentation above.
    Returns tensors with static shapes set.
    """
    image_out, label_out = tf.py_function(
        func=augment_py_func,
        inp=[image, label],
        Tout=[tf.float32, tf.float32]
    )

    # set static shape so TensorFlow knows the shapes downstream
    image_out.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    label_out.set_shape([IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES])

    return image_out, label_out

# ---- Dataset Generators ----
def train_gen():
    for x, y in zip(X_train, Y_train):
        yield x.astype(np.float32), y.astype(np.float32)

def valid_gen():
    for x, y in zip(X_valid, Y_valid):
        yield x.astype(np.float32), y.astype(np.float32)

def test_gen():
    for x, y in zip(X_test, Y_test):
        yield x.astype(np.float32), y.astype(np.float32)

# ---- Build Base Datasets ----
train_ds = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), tf.float32),
        tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), tf.float32),
    )
)

valid_ds = tf.data.Dataset.from_generator(
    valid_gen,
    output_signature=(
        tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), tf.float32),
        tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), tf.float32),
    )
)

test_ds = tf.data.Dataset.from_generator(
    test_gen,
    output_signature=(
        tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), tf.float32),
        tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), tf.float32),
    )
)

# ---- Multi-Output Label Mapper ----
def multi_output_mapper(x, y):
    return x, {
        'final_output': y,
        'out_3': y,
        'out_2': y,
        'out_1': y
    }

# ---- Apply pipeline (ORDER is IMPORTANT) ----
train_ds = train_ds.shuffle(num_train_samples)
train_ds = train_ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(multi_output_mapper, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.apply(tf.data.experimental.copy_to_device('/GPU:0'))
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = valid_ds.map(multi_output_mapper, num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE)
valid_ds = valid_ds.apply(tf.data.experimental.copy_to_device('/GPU:0'))
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# =============================================================================
# 3. MODEL ARCHITECTURE (same as original)
# =============================================================================
print("--- Defining ARU-GD Model ---")

def attention_block2d(x, g, inter_channel, i, data_format='channels_last'):
    theta_x = Conv2D(inter_channel, [2, 2], strides=[2, 2], data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], dilation_rate=i, data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(g)
    f = LeakyReLU(alpha=0.2)(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], dilation_rate=i, data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(f)
    sigm_psi_f = Activation(activation='sigmoid')(psi_f)
    rate = UpSampling2D(size=[2, 2])(sigm_psi_f)
    att_x = multiply([x, rate])
    return att_x


def res_block(x, nb_filters, strides, i):
    res_path = BatchNormalization()(x)
    res_path = LeakyReLU(alpha=0.2)(res_path)
    pool = MaxPooling2D(pool_size=(2, 2))(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', dilation_rate=i, strides=strides[1], kernel_regularizer=regularizers.l2(1e-5))(pool)
    res_path = BatchNormalization()(res_path)
    res_path = LeakyReLU(alpha=0.2)(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', dilation_rate=i, strides=strides[1], kernel_regularizer=regularizers.l2(1e-5))(res_path)
    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), dilation_rate=i, strides=strides[1], kernel_regularizer=regularizers.l2(1e-5))(pool)
    shortcut = BatchNormalization()(shortcut)
    res_path = Add()([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_regularizer=regularizers.l2(1e-5))(x)
    main_path = BatchNormalization()(main_path)
    main_path = LeakyReLU(alpha=0.2)(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_regularizer=regularizers.l2(1e-5))(main_path)
    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=regularizers.l2(1e-5))(x)
    shortcut = BatchNormalization()(shortcut)
    main_path = Add()([shortcut, main_path])
    to_decoder.append(main_path)
    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)], (1,1))
    to_decoder.append(main_path)
    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)], (2,2))
    to_decoder.append(main_path)
    main_path = res_block(main_path, [512, 512], [(2, 2), (1, 1)], (4,4))
    to_decoder.append(main_path)
    return to_decoder


def res_block_decoder(x, nb_filters, strides, i):
    res_path = BatchNormalization()(x)
    res_path = LeakyReLU(alpha=0.2)(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', dilation_rate=i, strides=strides[1], kernel_regularizer=regularizers.l2(1e-5))(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = LeakyReLU(alpha=0.2)(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', dilation_rate=i, strides=strides[1], kernel_regularizer=regularizers.l2(1e-5))(res_path)
    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[1], dilation_rate=i, kernel_regularizer=regularizers.l2(1e-5))(x)
    shortcut = BatchNormalization()(shortcut)
    res_path = Add()([shortcut, res_path])
    return res_path


def upsample_and_output(main_path, num_upsamples, num_classes):
    up = main_path
    for _ in range(num_upsamples):
        up = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-5))(up)
    return Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax')(up)


def decoder(x, from_encoder, dropout_rate):
    attention_path1 = attention_block2d(from_encoder[3], x, 256, (1,1))
    main_path1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    main_path1 = Concatenate()([main_path1, attention_path1])
    main_path1 = res_block_decoder(main_path1, [512, 512], [(1, 1), (1, 1)], (4,4))
    main_path1 = MonteCarloDropout(dropout_rate)(main_path1)
    out_1 = upsample_and_output(main_path1, 3, NUM_CLASSES)
    out_1 = tf.keras.layers.Lambda(lambda x: x, name='out_1')(out_1)


    attention_path2 = attention_block2d(from_encoder[2], main_path1, 128, (1,1))
    main_path2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(main_path1)
    main_path2 = Concatenate()([main_path2, attention_path2])
    main_path2 = res_block_decoder(main_path2, [256, 256], [(1, 1), (1, 1)], (2,2))
    main_path2 = MonteCarloDropout(dropout_rate)(main_path2)
    out_2 = upsample_and_output(main_path2, 2, NUM_CLASSES)
    out_2 = tf.keras.layers.Lambda(lambda x: x, name='out_2')(out_2)

    attention_path3 = attention_block2d(from_encoder[1], main_path2, 64, (1,1))
    main_path3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(main_path2)
    main_path3 = Concatenate()([main_path3, attention_path3])
    main_path3 = res_block_decoder(main_path3, [128, 128], [(1, 1), (1, 1)], (1,1))
    main_path3 = MonteCarloDropout(dropout_rate)(main_path3)
    out_3 = upsample_and_output(main_path3, 1, NUM_CLASSES)
    out_3 = tf.keras.layers.Lambda(lambda x: x, name='out_3')(out_3)


    attention_path4 = attention_block2d(from_encoder[0], main_path3, 32, (1,1))
    main_path4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(main_path3)
    main_path4 = Concatenate()([main_path4, attention_path4])
    main_path4 = res_block_decoder(main_path4, [64, 64], [(1, 1), (1, 1)], (1,1))
    main_path4 = MonteCarloDropout(dropout_rate)(main_path4)
    return main_path4, out_3, out_2, out_1


def aru_gd(input_shape, dropout_rate):
    inputs = Input(shape=input_shape)
    to_decoder = encoder(inputs)
    path = res_block(to_decoder[3], [1024, 1024], [(2, 2), (1, 1)], (8,8))
    final_out, out_3, out_2, out_1 = decoder(path, from_encoder=to_decoder, dropout_rate=dropout_rate)
    final_out = Conv2D(filters=NUM_CLASSES, kernel_size=(1, 1), activation='softmax', name='final_output')(final_out)
    return Model(inputs=inputs, outputs=[final_out, out_3, out_2, out_1])

# =============================================================================
# 4. LOSS FUNCTIONS AND METRICS
# =============================================================================
print("--- Defining Loss Functions and Metrics for KiTS23 ---")

def dice_coef_per_class(y_true, y_pred, class_id, smooth=1e-6):
    y_true_f = K.flatten(y_true[..., class_id])
    y_pred_f = K.flatten(y_pred[..., class_id])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_score(y_true, y_pred):
    d1 = dice_coef_per_class(y_true, y_pred, 1)
    d2 = dice_coef_per_class(y_true, y_pred, 2)
    d3 = dice_coef_per_class(y_true, y_pred, 3)
    return (d1 + d2 + d3) / 3.0


def weighted_dice_score(y_true, y_pred):
    d0 = dice_coef_per_class(y_true, y_pred, 0)
    d1 = dice_coef_per_class(y_true, y_pred, 1)
    d2 = dice_coef_per_class(y_true, y_pred, 2)
    d3 = dice_coef_per_class(y_true, y_pred, 3)

    weighted_sum_of_dices = (d0*LOSS_WEIGHTS_VEC[0] + d1*LOSS_WEIGHTS_VEC[1] +
                             d2*LOSS_WEIGHTS_VEC[2] + d3*LOSS_WEIGHTS_VEC[3])

    sum_of_weights = tf.reduce_sum(LOSS_WEIGHTS_VEC)
    return weighted_sum_of_dices / sum_of_weights


def weighted_dice_loss(y_true, y_pred):
    return 1 - weighted_dice_score(y_true, y_pred)


def weighted_log_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * tf.math.log(y_pred) * LOSS_WEIGHTS_VEC
    return tf.reduce_mean(-tf.reduce_sum(loss, -1))


def gen_dice_loss(y_true, y_pred):
    return weighted_dice_loss(y_true, y_pred) + weighted_log_loss(y_true, y_pred)

# =============================================================================
# 5. COMPILE AND TRAIN
# =============================================================================
print("--- Compiling Model ---")

tf.keras.backend.clear_session()
model = aru_gd(input_shape=INPUT_SHAPE, dropout_rate=DROPOUT_RATE)

losses = {
    'final_output': gen_dice_loss,
    'out_3': gen_dice_loss,
    'out_2': gen_dice_loss,
    'out_1': gen_dice_loss,
}
loss_weights = {"final_output": 1.0, "out_3": 0.25, "out_2": 0.25, "out_1": 0.25}

optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=loss_weights,
    metrics={'final_output': dice_score}
)

model.summary()

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    mode='min'
)

print("--- Starting Training ---")
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint],
    verbose=1
)

print(f"--- Training Finished. Best model saved to {MODEL_SAVE_PATH} ---")