import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


BASE_DIR   = "/mnt/c/Users/chloe/OneDrive/桌面/auto_angle"
TEST_DIR   = "/mnt/c/Users/chloe/OneDrive/桌面/auto_angle/test_data"
MODEL_PATH = os.path.join(BASE_DIR, "vessel_lumen_resnet50_unet.h5")
PRED_DIR   = os.path.join(BASE_DIR, "predictions_resnet50")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# --- Loss & Metric ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


# --- 模型（需與 resnet50_unet.py 一致） ---
def build_resnet50_unet(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)
    x3 = layers.Concatenate()([inputs, inputs, inputs])
    encoder = ResNet50(input_tensor=x3, include_top=False, weights=None)

    s1 = encoder.get_layer('conv1_relu').output
    s2 = encoder.get_layer('conv2_block3_out').output
    s3 = encoder.get_layer('conv3_block4_out').output
    s4 = encoder.get_layer('conv4_block6_out').output
    bridge = encoder.output

    def upsample_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x

    d1 = upsample_block(bridge, s4, 256)
    d2 = upsample_block(d1,     s3, 128)
    d3 = upsample_block(d2,     s2, 64)
    d4 = upsample_block(d3,     s1, 32)
    x  = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs, name="ResNet50_UNET")
    model.compile(optimizer=Adam(1e-4), loss=combined_loss, metrics=[dice_coef])
    return model


# --- 資料載入 ---
def load_test_data(test_dir, target_size=(224, 224)):
    img_dir  = os.path.join(test_dir, "test_images")
    mask_dir = os.path.join(test_dir, "test_masks")
    raw_images, raw_masks, filenames = [], [], []

    for filename in sorted(os.listdir(img_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path  = os.path.join(img_dir, filename)
        base_name = filename.rsplit('.', 1)[0]
        mask_path = os.path.join(mask_dir, f"{base_name}_label.png")

        if os.path.exists(mask_path):
            img  = cv2.imdecode(np.fromfile(img_path,  dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img.shape[:2]  != target_size: img  = cv2.resize(img,  target_size, interpolation=cv2.INTER_AREA)
            if mask.shape[:2] != target_size: mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            raw_images.append(img)
            raw_masks.append(mask)
            filenames.append(filename)
        else:
            print(f"⚠️ 找不到對應標記圖，跳過: {filename}")

    return np.array(raw_images), np.array(raw_masks), filenames


def preprocess(imgs, masks):
    X = (imgs.astype(np.float32) / 127.5) - 1.0
    Y = masks.astype(np.float32) / 255.0
    return np.expand_dims(X, -1), np.expand_dims(Y, -1)


def dice_per_image(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten().astype(np.float32)
    y_pred_f = y_pred.flatten().astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


if __name__ == "__main__":

    print("Loading test data...")
    raw_imgs, raw_masks, filenames = load_test_data(TEST_DIR)
    X_test, Y_test = preprocess(raw_imgs, raw_masks)
    print(f"Total: {len(raw_imgs)} images")

    print("Building model & loading weights...")
    model = build_resnet50_unet(input_shape=(224, 224, 1))
    model.load_weights(MODEL_PATH)

    preds = model.predict(X_test, verbose=0)

    print(f"\nPrediction stats:")
    print(f"  min = {preds.min():.6f}")
    print(f"  max = {preds.max():.6f}")
    print(f"  mean= {preds.mean():.6f}")
    print(f"  >0.5 ratio = {(preds > 0.5).mean():.4f}")

    os.makedirs(PRED_DIR, exist_ok=True)
    result_path = os.path.join(BASE_DIR, "test_resnet50_results.txt")
    dice_scores = []

    with open(result_path, "w", encoding="utf-8") as f:
        header = f"{'Filename':<40} {'Dice':>8}\n" + "-" * 55 + "\n"
        f.write(header)
        print(header, end="")

        for i, fname in enumerate(filenames):
            d = dice_per_image(Y_test[i], preds[i])
            dice_scores.append(d)
            line = f"{fname:<40} {d:>8.4f}\n"
            f.write(line)
            print(line, end="")

            mask = (preds[i] > 0.5).astype(np.uint8) * 255
            mask = np.squeeze(mask)
            cv2.imencode('.png', mask)[1].tofile(os.path.join(PRED_DIR, fname))

        dice_scores = np.array(dice_scores)
        stats = (
            "-" * 55 + "\n"
            f"Mean ± Std : {dice_scores.mean():.4f} ± {dice_scores.std():.4f}\n"
            f"Min / Max  : {dice_scores.min():.4f} / {dice_scores.max():.4f}\n"
            f"Count      : {len(dice_scores)}\n"
            f"Masks dir  : {PRED_DIR}\n"
        )
        f.write(stats)
        print(stats)

    print(f"Results saved to {result_path}")
