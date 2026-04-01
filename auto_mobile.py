import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.optimizers import Adam
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter

# --- 1.  (Dice & IoU) ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):

    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# --- 2. MobileNetV3 + UNET ---
def build_mobilenetv3_unet(input_shape=(224, 224, 1)): 

    inputs = layers.Input(shape=input_shape)
    
    #Encoder: MobileNetV3Small
    encoder = MobileNetV3Small(input_tensor=inputs, include_top=False, weights=None)
    
    # Skip Connection
    s1 = encoder.layers[4].output                                  # 112x112
    s2 = encoder.get_layer('expanded_conv_project_bn').output      # 56x56
    s3 = encoder.get_layer('expanded_conv_2_add').output           # 28x28
    s4 = encoder.get_layer('expanded_conv_5_add').output           # 14x14
    bridge = encoder.output                                        # 7x7

    # Decoder
    def upsample_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x

    d1 = upsample_block(bridge, s4, 256)
    d2 = upsample_block(d1, s3, 128)
    d3 = upsample_block(d2, s2, 64)
    d4 = upsample_block(d3, s1, 32)

    # 回復至原始大小並輸出 
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs, name="MobileNetV3_UNET")
    # 使用 Adam 優化器與二元交叉熵損失 
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', 
                  metrics=['accuracy', dice_coef, iou_coef])
    return model

# --- 3. 資料處理工具 (保留 Hsin-Yu 原本的血管軸搜尋邏輯) ---
def get_cropImg_axis(img):   
    # 透過亮度加總尋找血管壁位置
    row_sum = gaussian_filter(np.sum(img, axis=1), sigma=7)
    local_max = argrelextrema(row_sum, np.greater)[0]
    
    if len(local_max) < 2: return None
    
    # 抓取亮度最高的兩個峰值，代表上下壁 
    max_idx = np.argsort(row_sum[local_max])[-2:] 
    wallUp, wallDown = np.sort(local_max[max_idx])
    axis = (wallUp + wallDown) // 2  # 血管中心軸
    
    
    dist = wallDown - axis
    cropLineUp = max(0, axis - int(dist * 1.5))
    cropLineDown = min(img.shape[0], axis + int(dist * 1.5)) 
    
    cropImgFull = img[cropLineUp:cropLineDown, :]
    
    
    return cropImgFull, axis - cropLineUp, wallUp - cropLineUp, wallDown - cropLineUp

def load_data(path):
    images, masks = [], []
    for root, _, files in os.walk(path):
        for file in files:
            fullpath = os.path.join(root, file)
            if ".CRI" in file:
                img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)[50:500, 100:700]
                images.append(img)
            elif ".png" in file:
                mask = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)[50:500, 100:700]
                masks.append(mask)
    return np.array(images), np.array(masks)

def system_augmentation(imgs, masks):
    """每 5 度旋轉一次 (-60 到 60 度) """
    aug_imgs, aug_masks = [], []
    for i in range(len(imgs)):
        for angle in range(-60, 65, 5):
            h, w = imgs[i].shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            aug_imgs.append(cv2.warpAffine(imgs[i], M, (w, h)))
            aug_masks.append(cv2.warpAffine(masks[i], M, (w, h)))
    return np.array(aug_imgs), np.array(aug_masks)

def preprocess_images(imgs, masks, size=(224, 224)):
    """影像縮放與二次正規化"""
    X, Y = [], []
    for i in range(len(imgs)):
        img = cv2.resize(imgs[i], size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(masks[i], size, interpolation=cv2.INTER_NEAREST)
        # 正規化至 [-1, 1] 
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        X.append(img)
        Y.append(mask.astype(np.float32) / 255.0)
    return np.array(X).reshape(-1, size[0], size[1], 1), \
           np.array(Y).reshape(-1, size[0], size[1], 1)

if __name__ == "__main__":
    DATA_PATH = "drive/MyDrive/大三專題/100-IMT-ImagesCY/train"
    
    print("read data...")
    raw_imgs, raw_masks = load_data(DATA_PATH)
    
    print("vessel segmentation...")
    bot_imgs, bot_masks = [], []
    for i in range(len(raw_imgs)):
        res = get_cropImg_axis(raw_imgs[i])
        if res:
            bot_imgs.append(res[0])
            bot_masks.append(raw_masks[i][res[1]:res[2], :])
    
    #  Train (80%), Val (20%)
    split = int(len(bot_imgs) * 0.8)
    train_imgs, val_imgs = bot_imgs[:split], bot_imgs[split:]
    train_masks, val_masks = bot_masks[:split], bot_masks[split:]
    
    # data argumentation
    print("data argumentation...")
    train_imgs_aug, train_masks_aug = system_augmentation(train_imgs, train_masks)
    
    # Resize 224x224 + Normalization
    X_train, Y_train = preprocess_images(train_imgs_aug, train_masks_aug)
    X_val, Y_val = preprocess_images(val_imgs, val_masks)
    
    print("MobileNetV3 + UNET ...")
    model = build_mobilenetv3_unet(input_shape=(224, 224, 1))
    model.summary() # 預期參數約 3.4~3.5M 
    
    print("Start training...")
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                        epochs=50, batch_size=16)
    
    # plot training history
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
    plt.show()
    
    # 儲存模型
    model.save("mobilenet_unet_doppler.h5")
    print("model saved.")