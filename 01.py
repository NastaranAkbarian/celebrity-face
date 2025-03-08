import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# مسیر دیتاست
dataset_path = "archive (1)/Celebrity Faces Dataset/"

# تنظیمات پردازش داده‌ها
img_size = 224  # اندازه تصاویر ورودی
batch_size = 32  # تعداد داده‌های پردازش شده در هر مرحله

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


def predict_celeb(image_path, model, class_indices):
    # پردازش تصویر ورودی
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # اضافه کردن بعد دسته‌ای

    # پیش‌بینی
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # پیدا کردن نام فرد مشهور
    celeb_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_class)]
    return celeb_name

# بارگذاری مدل ذخیره شده
model = tf.keras.models.load_model("celebrity_face_recognition_model.h5")

# گرفتن تصویر از دوربین
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ذخیره تصویر
    user_img_path = "user_face.jpg"
    cv2.imwrite(user_img_path, frame)

    # پیش‌بینی نام فرد مشهور
    celeb_name = predict_celeb(user_img_path, model, train_generator.class_indices)

    # نمایش نتیجه روی تصویر
    cv2.putText(frame, f"Most Similar: {celeb_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # نمایش تصویر
    cv2.imshow("Celebrity Look-Alike", frame)

    # خروج با زدن کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
