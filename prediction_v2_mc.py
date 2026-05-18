"""
evaluate_kits23_arunet.py

Run:
    python evaluate_kits23_arunet.py

Requirements:
    - tensorflow (same major version used for training)
    - h5py
    - numpy
    - matplotlib
    - opencv-python (cv2) optional but used for nicer saving; matplotlib alone works too.
"""

import os
import random
import math
import numpy as np
import h5py
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from tensorflow.keras.layers import Dropout

# -----------------------
# CONFIG
# -----------------------
TEST_HDF5_PATH = "/home/tanmoyhazra/h5_kidney_dataset_monai/test_data.h5"
MODEL_PATH = "/home/tanmoyhazra/h5_kidney_dataset_monai/saved_model/kits23_arunet_model_v2.h5"
OUT_DIR = "/home/tanmoyhazra/h5_kidney_dataset_monai/predicted_images_v5_5/"

NUM_CLASSES = 4  # as used in training
IMG_CHANNELS = 1
NUM_SAMPLES_TO_SAVE = 50  # random samples to save (or fewer if test set smaller)
BATCH_SIZE = 16

# Monte Carlo Dropout parameters
T = 50  # Number of Monte Carlo samples
SENSITIVITY_FACTOR = 3.0  # Tune this to adjust red/green spread (higher = more red areas)

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Custom losses / metrics (must match training definitions)
# -----------------------
LOSS_WEIGHTS_VEC = tf.constant([0.5, 1.0, 3.0, 2.0], dtype=tf.float32)

def dice_coef_per_class(y_true, y_pred, class_id, smooth=1e-6):
    y_true_f = K.flatten(y_true[..., class_id])
    y_pred_f = K.flatten(y_pred[..., class_id])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_score(y_true, y_pred):
    # average of classes 1..3 (as in training script)
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
    return 1. - weighted_dice_score(y_true, y_pred)

def weighted_log_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * tf.math.log(y_pred) * LOSS_WEIGHTS_VEC
    return tf.reduce_mean(-tf.reduce_sum(loss, -1))

def gen_dice_loss(y_true, y_pred):
    return weighted_dice_loss(y_true, y_pred) + weighted_log_loss(y_true, y_pred)

# -----------------------
# Helper metric functions (numpy)
# -----------------------
def per_class_dice_np(y_true_onehot, y_pred_onehot, smooth=1e-6):
    # shapes: (H,W,C) or (N,H,W,C) ; implement single sample handling
    # will handle flattened samples
    y_true_f = y_true_onehot.reshape(-1, NUM_CLASSES).astype(np.float32)
    y_pred_f = y_pred_onehot.reshape(-1, NUM_CLASSES).astype(np.float32)
    dices = []
    for c in range(NUM_CLASSES):
        t = y_true_f[:, c]
        p = y_pred_f[:, c]
        inter = np.sum(t * p)
        den = np.sum(t) + np.sum(p)
        dice = (2. * inter + smooth) / (den + smooth)
        dices.append(dice)
    return np.array(dices)

def per_class_iou_np(y_true_labels, y_pred_labels, num_classes=NUM_CLASSES, smooth=1e-6):
    # inputs: label maps HxW or flattened; returns array of IoUs length num_classes
    y_true = y_true_labels.reshape(-1)
    y_pred = y_pred_labels.reshape(-1)
    ious = []
    for c in range(num_classes):
        t = (y_true == c).astype(np.uint8)
        p = (y_pred == c).astype(np.uint8)
        inter = np.sum(t & p)
        union = np.sum(t | p)
        iou = (inter + smooth) / (union + smooth)
        ious.append(iou)
    return np.array(ious)

def pixel_accuracy_np(y_true_labels, y_pred_labels):
    y_true = y_true_labels.reshape(-1)
    y_pred = y_pred_labels.reshape(-1)
    return float(np.mean(y_true == y_pred))

# -----------------------
# Color map for classes
# -----------------------
# You can choose colors per class (R,G,B) 0..1. Example:
CLASS_COLORS = np.array([
    [0.0, 0.0, 0.0],      # class 0 - background: black
    [1.0, 0.0, 0.0],      # class 1 - red
    [0.0, 1.0, 0.0],      # class 2 - green
    [0.0, 0.0, 1.0],      # class 3 - blue
], dtype=np.float32)

def label_to_color(label_map):
    # label_map: HxW integers
    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask = (label_map == c)
        color_img[mask] = CLASS_COLORS[c]
    return (color_img * 255).astype(np.uint8)

# -----------------------
# Load test data (entire file into memory)
# -----------------------
print("Loading test HDF5:", TEST_HDF5_PATH)
with h5py.File(TEST_HDF5_PATH, 'r') as f:
    X_test = f['X'][:]  # shape (N,H,W,channels) expected
    Y_test = f['Y'][:]

# Ensure float32 and shapes
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.float32)

num_test_samples = X_test.shape[0]
print(f"Test samples: {num_test_samples}")

# If Y_test is one-hot last axis equals NUM_CLASSES; if not, try to fix
if Y_test.ndim == 3:
    # maybe labels are single-channel ints: convert to one-hot
    print("Converting integer labels to one-hot (detected Y_test.ndim == 3)")
    y_int = Y_test.astype(np.int32)
    Y_test = tf.one_hot(y_int, NUM_CLASSES).numpy()
elif Y_test.shape[-1] != NUM_CLASSES:
    raise ValueError(f"Y_test last dim {Y_test.shape[-1]} != NUM_CLASSES {NUM_CLASSES}")

# If X_test has single channel but shape might be (N,H,W), fix to (N,H,W,1)
if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]

# Normalize inputs if needed (assumes training used raw floats between 0..255 or normalized - adapt if needed).
# If your training preprocessing scaled to [0,1], ensure same here:
if X_test.max() > 2.0:
    print("Assuming input range 0-255, scaling to [0,1]")
    X_test = X_test / 255.0

class MonteCarloDropout(Dropout):
    """Dropout layer that is always active (even during inference)."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # always apply dropout

# -----------------------
# Load model (custom objects)
# -----------------------
print("Loading model:", MODEL_PATH)
custom_objs = {
    'MonteCarloDropout': MonteCarloDropout,
    'gen_dice_loss': gen_dice_loss,
    'weighted_dice_loss': weighted_dice_loss,
    'weighted_log_loss': weighted_log_loss,
    'weighted_dice_score': weighted_dice_score,
    'dice_score': dice_score,
    # ensure Keras knows about LOSS_WEIGHTS_VEC symbol if used in any custom fn
    'LOSS_WEIGHTS_VEC': LOSS_WEIGHTS_VEC
}
model = load_model(MODEL_PATH, custom_objects=custom_objs, compile=False)
model.compile(optimizer='adam', loss=gen_dice_loss, metrics=[dice_score])
test_pred = model(X_test[:1], training=True)[0].numpy()
_, IMG_HEIGHT, IMG_WIDTH, _ = test_pred.shape
print("Model output:", IMG_HEIGHT, IMG_WIDTH)
model.summary()

# -----------------------
# Monte Carlo Predictions and Uncertainty
# -----------------------
print(f"\nGenerating {T} Monte Carlo samples for predictions and uncertainty...")
mc_predictions = np.zeros((T, num_test_samples, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), dtype=np.float32)

for t in range(T):
    print(f"\n  MC sample {t+1}/{T}...")
    for batch_start in range(0, num_test_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_test_samples)
        batch_x = X_test[batch_start:batch_end]

        pred_out = model(batch_x, training=True)
        pred_batch = pred_out[0].numpy()  # TAKE FIRST OUTPUT

        mc_predictions[t, batch_start:batch_end] = pred_batch

        print(f"    Processed batch {batch_end}/{num_test_samples}", end="\r")
    print()   # newline


# Compute mean probability and prediction
mean_prob = np.mean(mc_predictions, axis=0)  # (N, H, W, C)
pred_labels = np.argmax(mean_prob, axis=-1).astype(np.int32)  # (N, H, W)

# Compute predictive entropy for uncertainty
epsilon = 1e-7
mean_prob_clipped = np.clip(mean_prob, epsilon, 1 - epsilon)
entropy = -np.sum(mean_prob_clipped * np.log(mean_prob_clipped), axis=-1)  # (N, H, W)
max_entropy = np.log(NUM_CLASSES)
normalized_entropy = entropy / max_entropy  # Normalize 0-1

# Apply sensitivity
adjusted_uncertainty = np.clip(normalized_entropy * SENSITIVITY_FACTOR, 0, 1)
adjusted_uncertainty = np.nan_to_num(adjusted_uncertainty, nan=1.0, posinf=1.0, neginf=0.0)

# Resize ground truth to match prediction resolution
Y_test_resized = tf.image.resize(Y_test, (IMG_HEIGHT, IMG_WIDTH), method='nearest').numpy()
true_labels = np.argmax(Y_test_resized, axis=-1).astype(np.int32)

# -----------------------
# Compute metrics across test set (using MC mean predictions)
# -----------------------
per_sample_pixel_acc = []
per_sample_dice_per_class = []
per_sample_iou_per_class = []

for i in range(num_test_samples):
    acc = pixel_accuracy_np(true_labels[i], pred_labels[i])
    per_sample_pixel_acc.append(acc)

    # dice per class (on one sample)
    # convert one-hot for sample:
    y_true_onehot = Y_test_resized[i]
    pred_onehot = tf.one_hot(pred_labels[i], NUM_CLASSES).numpy().astype(np.float32)
    dice_per_class = per_class_dice_np(y_true_onehot, pred_onehot)
    per_sample_dice_per_class.append(dice_per_class)

    iou_per_class = per_class_iou_np(true_labels[i], pred_labels[i], num_classes=NUM_CLASSES)
    per_sample_iou_per_class.append(iou_per_class)

per_sample_pixel_acc = np.array(per_sample_pixel_acc)
per_sample_dice_per_class = np.array(per_sample_dice_per_class)  # shape (N, C)
per_sample_iou_per_class = np.array(per_sample_iou_per_class)

# Averages
mean_pixel_acc = float(np.mean(per_sample_pixel_acc))
mean_dice_per_class = np.mean(per_sample_dice_per_class, axis=0)
mean_iou_per_class = np.mean(per_sample_iou_per_class, axis=0)
mean_dice_all = float(np.mean(mean_dice_per_class))
mean_iou_all = float(np.mean(mean_iou_per_class))

# -----------------------
# Print summary
# -----------------------
print("\n=== Evaluation Summary ===")
print(f"Samples evaluated: {num_test_samples}")
print(f"Mean pixel accuracy (all pixels averaged): {mean_pixel_acc:.4f}")
for c in range(NUM_CLASSES):
    print(f"Class {c}: Mean Dice = {mean_dice_per_class[c]:.4f}, Mean IoU = {mean_iou_per_class[c]:.4f}")
print(f"Mean Dice (avg over classes): {mean_dice_all:.4f}")
print(f"Mean IoU (avg over classes): {mean_iou_all:.4f}")
print("==========================\n")

# Save numeric results to a CSV / text file
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_path = os.path.join(OUT_DIR, f"evaluation_summary_{ts}.txt")
with open(summary_path, "w") as fh:
    fh.write("Evaluation summary\n")
    fh.write(f"Date: {datetime.now().isoformat()}\n")
    fh.write(f"Test samples: {num_test_samples}\n")
    fh.write(f"Mean pixel accuracy: {mean_pixel_acc:.6f}\n")
    for c in range(NUM_CLASSES):
        fh.write(f"Class {c}: Mean Dice = {mean_dice_per_class[c]:.6f}, Mean IoU = {mean_iou_per_class[c]:.6f}\n")
    fh.write(f"Mean Dice (avg classes): {mean_dice_all:.6f}\n")
    fh.write(f"Mean IoU (avg classes): {mean_iou_all:.6f}\n")
print("class 0-background, class 1-healthy kidey red, class 2-tumour green, class 3-cyst blue")
print("Saved numeric summary to:", summary_path)

# -----------------------
# Save visual examples (random subset) with uncertainty
# -----------------------
# Custom colormap for uncertainty: green (confident/low unc) to red (uncertain/high unc)
cmap_unc = LinearSegmentedColormap.from_list('uncertainty', [
    (0.0, 'green'),   # Low uncertainty: green
    (0.5, 'yellow'),  # Medium
    (1.0, 'red')      # High uncertainty: red
])

num_to_save = min(NUM_SAMPLES_TO_SAVE, num_test_samples)
indices = random.sample(range(num_test_samples), num_to_save)
print(f"Saving {num_to_save} random prediction images (with uncertainty) to {OUT_DIR} ...")

for idx in indices:
    input_img = X_test[idx]  # H,W,1 or H,W if channel removed
    if input_img.ndim == 3 and input_img.shape[-1] == 1:
        input_disp = input_img[..., 0]
    else:
        input_disp = input_img.squeeze()

    gt_label = true_labels[idx]
    pred_label = pred_labels[idx]
    unc_map = adjusted_uncertainty[idx]  # H, W

    # colorize
    gt_color = label_to_color(gt_label)
    pred_color = label_to_color(pred_label)

    # overlay predicted mask on input (alpha)
    overlay = (0.6 * np.stack([input_disp, input_disp, input_disp], axis=-1) / (input_disp.max() + 1e-8) * 255.0).astype(np.uint8)
    blended_pred = (0.6 * pred_color.astype(np.float32) + 0.4 * overlay.astype(np.float32)).astype(np.uint8)

    # Compose a figure with 4 panels: input, GT, Predicted overlay, Uncertainty
    fig = plt.figure(figsize=(12, 3))
    ax1 = plt.subplot(1, 4, 1); ax1.imshow(input_disp, cmap='gray'); ax1.set_title('Input'); ax1.axis('off')
    ax2 = plt.subplot(1, 4, 2); ax2.imshow(gt_color); ax2.set_title('Ground Truth'); ax2.axis('off')
    ax3 = plt.subplot(1, 4, 3); ax3.imshow(blended_pred); ax3.set_title('Predicted (overlay)'); ax3.axis('off')
    ax4 = plt.subplot(1, 4, 4); ax4.imshow(unc_map, cmap=cmap_unc, vmin=0, vmax=1); ax4.set_title('Uncertainty'); ax4.axis('off')

    save_path = os.path.join(OUT_DIR, f"sample_{idx:04d}_acc{per_sample_pixel_acc[idx]:.3f}.jpg")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

print("Saved images for indices:", indices)
print("All done.")