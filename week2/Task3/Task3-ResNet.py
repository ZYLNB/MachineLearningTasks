import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# 定义图片大小
IMAGE_SIZE = (224, 224)

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

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# 加载预训练的ResNet50模型，不包括顶层的全连接层
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加自定义顶层
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 构建模型
model = Model(inputs=base_model.input, outputs=predictions)

# 冻结预训练模型的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型总结
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 保存模型
model.save('resnet50_food_classifier.h5')

from tensorflow.keras.models import load_model

# 加载模型
model = load_model('resnet50_food_classifier.h5')

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
