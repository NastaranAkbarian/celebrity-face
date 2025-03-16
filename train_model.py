import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# مسیر دقیق دیتاست
dataset_path = "C:/Users/ASUS/PycharmProjects/celebrity-face/archive (1)/Celebrity Faces Dataset/"

# بررسی وجود پوشه
if os.path.exists(dataset_path):
    print("Dataset path exists.")
else:
    print("Dataset path does not exist. Please check the path.")

# تنظیمات پردازش داده‌ها
img_size = 224  # اندازه تصاویر ورودی
batch_size = 32  # تعداد داده‌های پردازش شده در هر مرحله

# ایجاد دسته‌های آموزش و تست
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

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

# بررسی خروجی train_generator
print(f"Class indices: {train_generator.class_indices}")

# گرفتن تعداد کلاس‌ها (افراد مشهور)
num_classes = len(train_generator.class_indices)  # استخراج تعداد کلاس‌ها از train_generator
print(f"Number of classes: {num_classes}")

# استفاده از VGG16 به عنوان مدل پایه (بدون لایه‌های آخر)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

# فریز کردن لایه‌های از پیش آموزش‌دیده
for layer in base_model.layers:
    layer.trainable = False

# ایجاد مدل نهایی
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")  # خروجی به تعداد افراد مشهور
])

# کامپایل کردن مدل
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# نمایش ساختار مدل
model.summary()

# تعداد اپوک‌ها
epochs = 10

# آموزش مدل
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ذخیره مدل برای استفاده در آینده
model.save("celebrity_face_recognition_model.h5")
