import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import model


root = tk.Tk()
root.title("تشخیص شباهت چهره به افراد مشهور")
root.geometry("600x650")
root.configure(bg="pink")


header_label = tk.Label(root, text=" بازیگر شبیه خودتو پیدا کن! ", font=("Arial", 20, "bold"), pady=20, bg="purple",
                        fg="black")
header_label.pack()


image_label = tk.Label(root, bg="pink")
image_label.pack()


result_label = tk.Label(root, text="عکستو بده تا بگم شبیه کی هستی", font=("Arial", 18), bg="orange", fg="black")
result_label.pack()


def select_image():
    """ باز کردن فایل و پردازش تصویر """
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return

    # نمایش تصویر در GUI
    img = Image.open(file_path)
    img.thumbnail((250, 250))  # کاهش اندازه تصویر به صورت متناسب
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # پیش‌بینی و نمایش نتیجه
    celeb_name = model.predict_celeb(file_path)  # فراخوانی تابع از model.py
    result_label.config(text=f"شبیه‌ترین فرد: {celeb_name}")


def start_camera():
    """ فعال‌سازی دوربین و نمایش پیش‌نمایش برای گرفتن عکس """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_label.config(text="خطا در باز کردن دوربین!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("دوربین - برای گرفتن عکس کلید Space را بزنید", frame)
        key = cv2.waitKey(1)

        if key == 32:  # Space key
            file_path = "user_face.jpg"
            cv2.imwrite(file_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            process_captured_image(file_path)
            break
        elif key == 27:  # Escape key
            cap.release()
            cv2.destroyAllWindows()
            break


def process_captured_image(file_path):
    """ پردازش و نمایش تصویر گرفته شده """
    img = Image.open(file_path)
    img.thumbnail((250, 250))  # کاهش اندازه تصویر به صورت متناسب
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # پیش‌بینی و نمایش نتیجه
    celeb_name = model.predict_celeb(file_path)  # استفاده از model.py
    result_label.config(text=f"شبیه‌ترین فرد: {celeb_name}")


# دکمه‌های GUI
btn_select = tk.Button(root, text="📂 انتخاب تصویر از گالری", command=select_image, bg="white", fg="black")
btn_select.pack(pady=10)

btn_capture = tk.Button(root, text="📸 گرفتن عکس از دوربین", command=start_camera, bg="white", fg="black")
btn_capture.pack(pady=10)

# اجرای برنامه
root.mainloop()
