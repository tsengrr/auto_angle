import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# --- 評估指標 & Loss ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# --- 模型 ---
def build_mobilenetv3_unet(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)
    x3 = layers.Concatenate()([inputs, inputs, inputs])
    encoder = MobileNetV3Large(input_tensor=x3, include_top=False, weights='imagenet', include_preprocessing=False)

    s1 = encoder.get_layer('re_lu').output                # 112x112
    s2 = encoder.get_layer('expanded_conv_2_add').output  # 56x56
    s3 = encoder.get_layer('expanded_conv_5_add').output  # 28x28
    s4 = encoder.get_layer('expanded_conv_11_add').output # 14x14
    bridge = encoder.output                               # 7x7

    def upsample_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x

    d1 = upsample_block(bridge, s4, 256)
    d2 = upsample_block(d1, s3, 128)
    d3 = upsample_block(d2, s2, 64)
    d4 = upsample_block(d3, s1, 32)
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs, name="MobileNetV3_UNET")
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=combined_loss,
                  metrics=[dice_coef])
    return model


# --- 資料載入 ---
def load_and_match_data(path):
    images, masks, filenames = [], [], []
    img_dir  = os.path.join(path, "images")
    mask_dir = os.path.join(path, "masks")

    for filename in sorted(os.listdir(img_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path  = os.path.join(img_dir, filename)
        base_name = filename.rsplit('.', 1)[0]
        mask_path = os.path.join(mask_dir, f"{base_name}_label.png")

        if os.path.exists(mask_path):
            img  = cv2.imdecode(np.fromfile(img_path,  dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img.shape[:2]  != (224, 224): img  = cv2.resize(img,  (224, 224))
            if mask.shape[:2] != (224, 224): mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            images.append(img)
            masks.append(mask)
            filenames.append(base_name)
        else:
            print(f"⚠️ 找不到對應的標記圖，已跳過: {filename}")

    return np.array(images), np.array(masks), filenames


def clip_level_split(imgs, masks, filenames, val_clips=('data4', 'data6')):
    """固定 val = data4 + data6，其餘全部丟 train"""
    clip_ids = [fname.rsplit('_', 1)[0] for fname in filenames]
    unique_clips = sorted(set(clip_ids))

    val_set = set(val_clips)
    train_clips = [c for c in unique_clips if c not in val_set]

    train_idx = [i for i, cid in enumerate(clip_ids) if cid not in val_set]
    val_idx   = [i for i, cid in enumerate(clip_ids) if cid in val_set]

    print(f"clips: {len(unique_clips)} total → {len(train_clips)} train / {len(val_set)} val ({sorted(val_set)})")
    print(f"frames: {len(train_idx)} train / {len(val_idx)} val")

    return imgs[train_idx], masks[train_idx], imgs[val_idx], masks[val_idx]

def system_augmentation(imgs, masks):
    """每 5 度旋轉一次 (-60 到 60 度)"""
    aug_imgs, aug_masks = [], []
    for i in range(len(imgs)):
        for angle in range(-60, 65, 5):
            h, w = imgs[i].shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            aug_imgs.append(cv2.warpAffine(imgs[i], M, (w, h)))
            aug_masks.append(cv2.warpAffine(masks[i], M, (w, h), flags=cv2.INTER_NEAREST))
    return np.array(aug_imgs), np.array(aug_masks)



def preprocess_for_model(imgs, masks):
    X = (imgs.astype(np.float32) / 255.0 - 0.5) / 0.5
    Y = masks.astype(np.float32) / 255.0
    return np.expand_dims(X, -1), np.expand_dims(Y, -1)


if __name__ == "__main__":

    DATA_PATH  = "/mnt/c/Users/chloe/OneDrive/桌面/auto_angle"
    MODEL_PATH = "/mnt/c/Users/chloe/OneDrive/桌面/auto_angle/vessel_lumen_mobilenet_large_unet.h5"

    print("load data...")
    raw_imgs, raw_masks, filenames = load_and_match_data(DATA_PATH)

    train_imgs, train_masks, val_imgs, val_masks = clip_level_split(
        raw_imgs, raw_masks, filenames
    )

    print(f"train frames: {len(train_imgs)}")
    X_aug, Y_aug = system_augmentation(train_imgs, train_masks)
    X_train, Y_train = preprocess_for_model(X_aug, Y_aug)
    del X_aug, Y_aug
    X_val, Y_val = preprocess_for_model(val_imgs, val_masks)
    print(f"augmented train size: {len(X_train)}")

    print("MobileNetV3 + U-Net...")
    model = build_mobilenetv3_unet(input_shape=(224, 224, 1))

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_dice_coef',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_dice_coef',
        mode='max',
        patience=20,
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_dice_coef',
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    print("Training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=150,
        batch_size=16,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 訓練歷程視覺化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['dice_coef'], label='Train Dice')
    plt.plot(history.history['val_dice_coef'], label='Val Dice')
    plt.title('Dice Score')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig("/mnt/c/Users/chloe/OneDrive/桌面/auto_angle/training_history_large.png", bbox_inches='tight')
    plt.show()
