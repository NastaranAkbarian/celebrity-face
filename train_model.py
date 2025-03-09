import os
import numpy as np
import cv2
import tensorflow as tf
from keras.src.backend.numpy.nn import batch_normalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import json  # برای ذخیره‌سازی class_indices
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json


dataset_path = "archive (1)/Celebrity Faces Dataset/"


img_size = 224
batch_size = 32

# ایجاد دسته‌های آموزش و تست
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# گرفتن تعداد کلاس‌ها (افراد مشهور)
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# ذخیره‌سازی اطلاعات کلاس‌ها در یک فایل JSON
with open("class_indices.json", "w") as json_file:
    json.dump(train_generator.class_indices, json_file)

model = Sequential([
    # لایه‌های کانولوشن
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # لایه فلَت کردن برای انتقال به لایه‌های Dense
    Flatten(),

    # لایه‌های Dense برای طبقه‌بندی
    Dense(512, activation='relu'),
    Dropout(0.5),  # استفاده از Dropout برای جلوگیری از overfitting
    Dense(num_classes, activation='softmax')  # خروجی به تعداد کلاس‌ها
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.summary()


epochs = 20


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)


model.save("celebrity_face_recognition_model.h5")
