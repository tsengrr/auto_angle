import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K


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


def load_test_data(test_dir, target_size=(224, 224)):
    """read testing data"""
    img_dir = os.path.join(test_dir, "images")
    mask_dir = os.path.join(test_dir, "masks")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError("cannot find test_data/images or test_data/masks")

    raw_images, raw_masks, filenames = [], [], []
    file_list = sorted(os.listdir(img_dir))
    
    for filename in file_list:
        
        # 防呆：確保只處理圖片檔，避免讀到 Mac 的 .DS_Store 等系統隱藏檔
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(img_dir, filename)
        
        # 【關鍵修改】透過原圖檔名，精準預測並組合出 Mask 的檔名
        # rsplit('.', 1)[0] 會從右邊切開第一個小數點，取出純檔名
        base_name = filename.rsplit('.', 1)[0]
        mask_filename = f"{base_name}_label.png"
        
        mask_path = os.path.join(mask_dir, mask_filename)
        
        # 檢查對應的 _label.png 存不存在
        if os.path.exists(mask_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # resize to 224*224
            if img.shape[:2] != target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            # Mask 必須維持 INTER_NEAREST
            if mask.shape[:2] != target_size:
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                
            raw_images.append(img)
            raw_masks.append(mask)
            
            # 這裡我們將 filenames 陣列保留「原圖的檔名 (filename)」
            # 這樣在執行 test.py 最後儲存預測結果時，輸出的 png 名字才會跟原圖一模一樣
            filenames.append(filename) 
        else:
            # 測試資料有缺漏時印出警告，方便除錯
            print(f"⚠️ 測試集中找不到對應的標記圖，已跳過: {filename} (預期應有: {mask_filename})")
            
    return np.array(raw_images), np.array(raw_masks), filenames

def preprocess_for_inference(imgs, masks):
    """Normalized to [0,1]"""
    X = (imgs.astype(np.float32) / 255.0 - 0.5) / 0.5
    Y = masks.astype(np.float32) / 255.0
    return np.expand_dims(X, axis=-1), np.expand_dims(Y, axis=-1)


if __name__ == "__main__":
    MODEL_PATH = "vessel_lumen_mobilenet_small_unet.h5"  
    TEST_DIR = "test_data"                         
    OUTPUT_DIR = "test_results"                    

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("reading testing data...")
    raw_imgs, raw_masks, filenames = load_test_data(TEST_DIR)
    print(f"total {len(raw_imgs)} images")

    X_test, Y_test = preprocess_for_inference(raw_imgs, raw_masks)

    print(f"loading model{MODEL_PATH} ...")
    model = load_model(MODEL_PATH, custom_objects={'dice_coef': dice_coef, 'iou_coef': iou_coef})

    print("\nTest Score...")
    results = model.evaluate(X_test, Y_test, verbose=0)
    print("="*30)
    print(f"Test Loss (BCE) : {results[0]:.4f}")
    print(f"Test Accuracy   : {results[1]:.4f}")
    print(f"Test Dice Score : {results[2]:.4f}")  # > 0.9 good good
    print("="*30)

    print("\n generating predicted mask...")
    predictions = model.predict(X_test)

    # 建立專門存放 prediction 的資料夾
    PRED_DIR = "predictions_output_small"
    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)

    for i in range(len(raw_imgs)):
        filename = filenames[i] 

        pred_mask = (predictions[i] > 0.5).astype(np.uint8) * 255
        
        pred_mask = np.squeeze(pred_mask) 

        save_path = os.path.join(PRED_DIR, filename)
        cv2.imwrite(save_path, pred_mask)

    print(f"\npredicted mask saved")