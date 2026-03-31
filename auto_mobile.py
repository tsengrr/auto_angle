import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
import matplotlib.pyplot as plt
import numpy as np

def build_mobilenetv3_unet(input_shape=(224, 224, 1)):
    """
    建構基於 MobileNetV3 的 UNET 影像分割模型
    """
    inputs = layers.Input(shape=input_shape)

    # --- 1. 編碼器 Encoder ---
    encoder = MobileNetV3Small(input_tensor=inputs, 
                               input_shape=input_shape, 
                               include_top=False, 
                               weights=None)

    # --- 2. 擷取特徵圖 (Skip Connections) ---
    s1 = encoder.layers[4].output      
    s2 = encoder.get_layer('expanded_conv_project_bn').output      
    s3 = encoder.get_layer('expanded_conv_2_add').output     
    s4 = encoder.get_layer('expanded_conv_5_add').output     
    bridge = encoder.output                                   

    # --- 3. 解碼器 Decoder (5次上取樣) ---
    x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bridge)
    x = layers.Concatenate()([x, s4])
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Concatenate()([x, s3])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Concatenate()([x, s2])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Concatenate()([x, s1])
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    # --- 4. 輸出層 ---
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    return models.Model(inputs, outputs, name='MobileNetV3Small_UNET')

def train_and_plot():
    # 1. 準備模擬資料 (請在實際應用時替換成你的超音波影像矩陣)
    print("正在準備 Train / Test 資料...")
    X_train = np.random.rand(10, 224, 224, 1)  # 10張訓練圖
    Y_train = np.random.randint(0, 2, size=(10, 224, 224, 1)) # 10張訓練標註
    
    X_test = np.random.rand(4, 224, 224, 1)    # 4張測試圖
    Y_test = np.random.randint(0, 2, size=(4, 224, 224, 1))   # 4張測試標註

    # 2. 建構與編譯模型
    model = build_mobilenetv3_unet()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']) # 這裡紀錄準確率

    # 3. 執行訓練並儲存歷程 (History)
    print("開始訓練模型...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test), # 傳入測試資料集
        epochs=10,                        # 測試跑 10 個 Epoch
        batch_size=2,
        verbose=1                         # 顯示訓練進度條
    )

    # 4. 繪製 Accuracy 曲線
    print("繪製訓練曲線...")
    plt.figure(figsize=(8, 5))
    
    # 取出 history 字典裡的 acc 與 val_acc
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Test (Validation) Accuracy', linewidth=2, linestyle='--')
    
    plt.title('Model Accuracy per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 顯示圖表
    plt.show()

# 執行主程式
if __name__ == '__main__':
    train_and_plot()