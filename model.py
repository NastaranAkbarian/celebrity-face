import cv2
import numpy as np
import tensorflow as tf
import json

# بارگذاری مدل ذخیره‌شده
model = tf.keras.models.load_model("celebrity_face_recognition_model.h5")

# بارگذاری نام کلاس‌ها
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_indices = {v: k for k, v in class_indices.items()}  # معکوس کردن دیکشنری

# اندازه تصویر ورودی
img_size = 224

def predict_celeb(image_path):
    """ دریافت تصویر و پیش‌بینی نام فرد مشهور """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # نرمال‌سازی
    img = np.expand_dims(img, axis=0)  # اضافه کردن بعد دسته‌ای

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    celeb_name = class_indices[predicted_class]

    return celeb_name
