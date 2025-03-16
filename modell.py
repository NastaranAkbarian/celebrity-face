import numpy as np
import tensorflow as tf
import cv2
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# بارگذاری مدل ذخیره‌شده
model = tf.keras.models.load_model("celebrity_face_recognition_cnn_model.h5")

# بارگذاری اطلاعات کلاس‌ها از فایل JSON
with open("class_indices.json", "r") as json_file:
    class_indices = json.load(json_file)

# تبدیل دیکشنری class_indices به یک لیست برای دسترسی به نام‌ها
class_labels = {v: k for k, v in class_indices.items()}  # معکوس کردن دیکشنری

# تابع پیش‌بینی تصویر
def predict_image(image_path):
    img_size = 224  # اندازه‌ای که مدل با آن آموزش دیده است

    # بارگذاری و پیش‌پردازش تصویر
    img = load_img(image_path, target_size=(img_size, img_size))  # تغییر اندازه تصویر
    img_array = img_to_array(img) / 255.0  # نرمال‌سازی به بازه 0 تا 1
    img_array = np.expand_dims(img_array, axis=0)  # اضافه کردن بعد برای مدل

    # پیش‌بینی با مدل
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # گرفتن کلاس با بالاترین احتمال
    predicted_label = class_labels[predicted_class]  # تبدیل شماره کلاس به نام فرد مشهور

    # نمایش تصویر همراه با پیش‌بینی
    img_cv = cv2.imread(image_path)  # خواندن تصویر با OpenCV
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # تبدیل رنگ برای نمایش صحیح در Matplotlib

    plt.figure(figsize=(6, 6))
    plt.imshow(img_cv)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label}", fontsize=14, fontweight="bold", color="red")
    plt.show()

# آزمایش روی یک تصویر نمونه
image_path = "path_to_test_image.jpg"  # مسیر تصویر تست
predict_image(image_path)
