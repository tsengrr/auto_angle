import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import random

SEED = 123
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)


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
def double_conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_pure_unet(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = double_conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = double_conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = double_conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = double_conv_block(p3, 256)
    c4 = layers.SpatialDropout2D(0.3)(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bridge
    b5 = double_conv_block(p4, 512)
    b5 = layers.SpatialDropout2D(0.3)(b5)

    # Decoder
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(b5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = double_conv_block(u6, 256)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = double_conv_block(u7, 128)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = double_conv_block(u8, 64)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = double_conv_block(u9, 32)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = models.Model(inputs, outputs, name="Pure_UNET")
    model.compile(optimizer=Adam(learning_rate=5e-4),
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


def clip_level_split(imgs, masks, filenames, val_clips=('data1',)):
    clip_ids = [fname.split('_', 1)[0] for fname in filenames]
    unique_clips = sorted(set(clip_ids))
    val_set = set(val_clips)
    train_clips = [c for c in unique_clips if c not in val_set]
    train_idx = [i for i, cid in enumerate(clip_ids) if cid not in val_set]
    val_idx   = [i for i, cid in enumerate(clip_ids) if cid in val_set]
    print(f"clips: {len(unique_clips)} total → {len(train_clips)} train / {len(val_set)} val ({sorted(val_set)})")
    print(f"frames: {len(train_idx)} train / {len(val_idx)} val")
    return imgs[train_idx], masks[train_idx], imgs[val_idx], masks[val_idx]


_AUG_ANGLES = [-20, -10, 0, 10, 20]

def system_augmentation(imgs, masks):
    """
    5 角度 × 3 幾何 × 4 強度 = 60× 放大
    幾何：原圖、水平翻轉、非等比例拉伸
    強度：高斯模糊、加法噪點、低對比、probe dropout
    """
    aug_imgs, aug_masks = [], []

    for i in range(len(imgs)):
        img, mask = imgs[i], masks[i]
        h, w = img.shape[:2]

        for angle in _AUG_ANGLES:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            r_img  = cv2.warpAffine(img,  M, (w, h))
            r_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

            # 幾何 A：原旋轉
            geo_a_img,  geo_a_mask  = r_img, r_mask

            # 幾何 B：水平翻轉
            geo_b_img  = cv2.flip(r_img, 1)
            geo_b_mask = cv2.flip(r_mask, 1)

            # 幾何 C：非等比例拉伸 → center crop（只放大不縮小）
            sy = random.uniform(1.0, 1.3)
            sx = random.uniform(1.0, 1.15)
            nh, nw = max(int(h * sy), h), max(int(w * sx), w)
            s_img  = cv2.resize(r_img,  (nw, nh), interpolation=cv2.INTER_LINEAR)
            s_mask = cv2.resize(r_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            y0 = (nh - h) // 2;  x0 = (nw - w) // 2
            geo_c_img  = s_img [y0:y0 + h, x0:x0 + w]
            geo_c_mask = s_mask[y0:y0 + h, x0:x0 + w]

            for g_img, g_mask in [(geo_a_img, geo_a_mask),
                                   (geo_b_img, geo_b_mask),
                                   (geo_c_img, geo_c_mask)]:

                # 強度 1：高斯模糊
                aug_imgs.append(cv2.GaussianBlur(g_img, (5, 5), 0))
                aug_masks.append(g_mask)

                # 強度 2：加法高斯噪點
                noise = np.random.normal(0, 15, g_img.shape).astype(np.int32)
                aug_imgs.append(np.clip(g_img.astype(np.int32) + noise, 0, 255).astype(np.uint8))
                aug_masks.append(g_mask)

                # 強度 3：低對比度
                alpha_c = random.uniform(0.5, 0.8)
                beta_b  = random.randint(10, 30)
                aug_imgs.append(cv2.convertScaleAbs(g_img, alpha=alpha_c, beta=beta_b))
                aug_masks.append(g_mask)

                # 強度 4：probe dropout（垂直遮蔽帶）
                dw  = random.randint(10, 40)
                dx  = random.randint(0, w - dw)
                di  = random.uniform(0.1, 0.3)
                drop_mask = np.ones((h, w), dtype=np.float32)
                drop_mask[:, dx:dx + dw] = di
                drop_mask = cv2.GaussianBlur(drop_mask, (15, 1), 0)
                aug_imgs.append(np.clip(g_img.astype(np.float32) * drop_mask, 0, 255).astype(np.uint8))
                aug_masks.append(g_mask)

    return np.array(aug_imgs), np.array(aug_masks)


def preprocess_for_model(imgs, masks):
    X = (imgs.astype(np.float32) / 255.0 - 0.5) / 0.5
    Y = masks.astype(np.float32) / 255.0
    return np.expand_dims(X, -1), np.expand_dims(Y, -1)


if __name__ == "__main__":

    DATA_PATH  = "/mnt/c/Users/chloe/OneDrive/桌面/auto_angle"
    MODEL_PATH = f"/mnt/c/Users/chloe/OneDrive/桌面/auto_angle/vessel_lumen_pure_unet_aug5a3g4i_seed{SEED}.h5"

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

    print("Pure U-Net...")
    model = build_pure_unet(input_shape=(224, 224, 1))
    model.summary()

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
        patience=30,
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
        epochs=100,
        batch_size=16,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

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
    plt.savefig(f"/mnt/c/Users/chloe/OneDrive/桌面/auto_angle/training_history_pure_unet_aug5a3g4i_seed{SEED}.png",
                bbox_inches='tight')
    plt.show()
