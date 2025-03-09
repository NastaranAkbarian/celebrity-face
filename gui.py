import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import model  # ایمپورت فایل model.py

# ایجاد پنجره اصلی
root = tk.Tk()
root.title("تشخیص شباهت چهره به افراد مشهور")
root.geometry("600x500")

# نمایش تصویر
image_label = tk.Label(root)
image_label.pack()

# نمایش نتیجه
result_label = tk.Label(root, text="منتظر دریافت تصویر...", font=("Arial", 14))
result_label.pack()

def select_image():
    """ باز کردن فایل و پردازش تصویر """
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return

    # نمایش تصویر در GUI
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # پیش‌بینی و نمایش نتیجه
    celeb_name = model.predict_celeb(file_path)  # فراخوانی تابع از model.py
    result_label.config(text=f"شبیه‌ترین فرد: {celeb_name}")

def capture_image():
    """ گرفتن تصویر از دوربین و پیش‌بینی فرد مشهور """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        result_label.config(text="خطا در دریافت تصویر!")
        return

    file_path = "user_face.jpg"
    cv2.imwrite(file_path, frame)

    # نمایش تصویر در GUI
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # پیش‌بینی و نمایش نتیجه
    celeb_name = model.predict_celeb(file_path)  # استفاده از model.py
    result_label.config(text=f"شبیه‌ترین فرد: {celeb_name}")

# دکمه‌های GUI
btn_select = tk.Button(root, text="📂 انتخاب تصویر از گالری", command=select_image)
btn_select.pack(pady=10)

btn_capture = tk.Button(root, text="📸 گرفتن عکس از دوربین", command=capture_image)
btn_capture.pack(pady=10)

# اجرای برنامه
root.mainloop()
