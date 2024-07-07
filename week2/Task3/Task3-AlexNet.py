import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# 定义图片大小
IMAGE_SIZE = (227, 227)

# 读取数据
def load_dataset(path, has_labels=True):
    images = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 获取标签（仅用于训练和验证集）
            if has_labels:
                label = int(filename.split('_')[0])
                labels.append(label)
            # 读取图像
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
    images = np.array(images)
    if has_labels:
        labels = np.array(labels)
        return images, labels
    else:
        return images

# 加载训练集、验证集和测试集
train_path = './food11/training'
validation_path = './food11/validation'
test_path = './food11/test'

X_train, y_train = load_dataset(train_path)
X_val, y_val = load_dataset(validation_path)
X_test = load_dataset(test_path, has_labels=False)

# 数据归一化
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# 转换标签为one-hot编码
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_alexnet(input_shape, num_classes):
    model = Sequential()
    
    # 第一层卷积层
    model.add(Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((3, 3), strides=2))
    
    # 第二层卷积层
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=2))
    
    # 第三、四、五层卷积层
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=2))
    
    # 全连接层
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# 定义输入形状
input_shape = (227, 227, 3)

# 创建模型
model = create_alexnet(input_shape, num_classes)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型总结
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 保存模型
model.save('alexnet_food_classifier.h5')

from tensorflow.keras.models import load_model

# 加载模型
model = load_model('alexnet_food_classifier.h5')

# 预测测试集
predictions = model.predict(X_test)

# 将预测结果转换为类别标签
predicted_classes = np.argmax(predictions, axis=1)

# 保存预测结果
test_filenames = os.listdir(test_path)

with open('test_predictions.txt', 'w') as f:
    for filename, predicted_class in zip(test_filenames, predicted_classes):
        f.write(f'{filename}: {predicted_class}\n')

print("Predictions saved to test_predictions.txt")
