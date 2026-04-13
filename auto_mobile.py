import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# --- 1. 評估指標定義 (Metrics) ---
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

def build_mobilenetv3_unet(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)
    
    encoder = MobileNetV3Small(input_tensor=inputs, include_top=False, weights=None)
    
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

    d1 = upsample_block(bridge, s4, 256) # 14x14
    d2 = upsample_block(d1, s3, 128)     # 28x28
    d3 = upsample_block(d2, s2, 64)      # 56x56
    d4 = upsample_block(d3, s1, 32)      # 112x112

    # 224x224
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs, name="MobileNetV3_UNET")
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', dice_coef, iou_coef])
    return model

def load_and_match_data(path):
    images, masks = [], []
    img_dir = os.path.join(path, "images") # raw data dir path 
    mask_dir = os.path.join(path, "masks") # labeled data dir path
    
    file_list = sorted(os.listdir(img_dir))
    for filename in file_list:
        
        # 防呆：確保只處理圖片檔，避免讀到 Mac 的 .DS_Store 等系統隱藏檔
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(img_dir, filename)
        
        # 【關鍵修改】透過原圖檔名，精準預測並組合出 Mask 的檔名
        # rsplit('.', 1)[0] 會從右邊切開第一個小數點，取出純檔名 (例如 "data1_5.83s")
        base_name = filename.rsplit('.', 1)[0]
        mask_filename = f"{base_name}_label.png"
        
        mask_path = os.path.join(mask_dir, mask_filename) 
        
        # 檢查這個 _label.png 存不存在
        if os.path.exists(mask_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # make sure if is 224x224
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
            # 注意：Mask 必須維持 INTER_NEAREST 避免產生灰階雜訊
            if mask.shape[:2] != (224, 224):
                mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                
            images.append(img)
            masks.append(mask)
        else:
            # 如果發現有原圖沒標記，印出警告但不中斷程式
            print(f"⚠️ 找不到對應的標記圖，已跳過: {filename} (預期應有: {mask_filename})")
            
    return np.array(images), np.array(masks)

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
    """Normalized to [-1, 1]"""
    X = (imgs.astype(np.float32) / 255.0 - 0.5) / 0.5
    Y = masks.astype(np.float32) / 255.0
    return np.expand_dims(X, axis=-1), np.expand_dims(Y, axis=-1)

if __name__ == "__main__":

    DATA_PATH = "drive/MyDrive/path_to_your_dataset" # need to change
    
    print("load data...")
    raw_imgs, raw_masks = load_and_match_data(DATA_PATH)
    
    # 2. 訓練/驗證集切分 (8:2)
    indices = np.arange(len(raw_imgs))
    np.random.shuffle(indices) #切！
    split = int(len(indices) * 0.8)
    
    train_idx, val_idx = indices[:split], indices[split:]
    train_imgs, train_masks = raw_imgs[train_idx], raw_masks[train_idx]
    val_imgs, val_masks = raw_imgs[val_idx], raw_masks[val_idx]
    
    # 3. data augmentation
    print(f"the number of raw data: {len(train_imgs)}")
    X_train_aug, Y_train_aug = system_augmentation(train_imgs, train_masks)
    print(f"final number: {len(X_train_aug)}")
    
    # 4. Normalize
    X_train, Y_train = preprocess_for_model(X_train_aug, Y_train_aug)
    X_val, Y_val = preprocess_for_model(val_imgs, val_masks)
    
    # 5. 硬Train一發
    print("MobileNetV3 + UNET...")
    model = build_mobilenetv3_unet(input_shape=(224, 224, 1))
    
    checkpoint = ModelCheckpoint(
    "vessel_lumen_mobilenet_small_unet.h5", 
    monitor='val_dice_coef', 
    mode='max', 
    save_best_only=True, 
    verbose=1
    )
       
    print("Training...")
    history = model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val), 
        epochs=50, 
        batch_size=16,
        callbacks=[checkpoint] # 把規則塞進去
    )
    
    # 6. 訓練歷程視覺化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['dice_coef'], label='Train Dice')
    plt.plot(history.history['val_dice_coef'], label='Val Dice')
    plt.title('Dice Score')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Binary Crossentropy Loss')
    plt.legend()
    plt.savefig("training_history_small.png", bbox_inches='tight')
    plt.show()
    
    