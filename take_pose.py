import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import glob

def create_directories(base_path):
    os.makedirs(os.path.join(base_path, 'good'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'bad'), exist_ok=True)

def list_available_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(index)
        cap.release()
        index += 1
    return cameras

class PostureCaptureApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Posture Capture")
        self.master.geometry("1200x800")

        self.camera_var = tk.StringVar()
        self.pose_type_var = tk.StringVar(value="good")
        self.capture_count = 0
        self.max_captures = 50
        self.dataset_path = "posture_dataset"

        self.setup_ui()
        self.cameras = list_available_cameras()
        self.populate_camera_list()
        self.update_dataset_info()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Camera selection
        ttk.Label(left_frame, text="Select Camera:").pack(pady=5)
        self.camera_combo = ttk.Combobox(left_frame, textvariable=self.camera_var)
        self.camera_combo.pack(pady=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)

        # Pose type selection
        ttk.Label(left_frame, text="Pose Type:").pack(pady=5)
        ttk.Radiobutton(left_frame, text="Good", variable=self.pose_type_var, value="good", command=self.update_dataset_preview).pack()
        ttk.Radiobutton(left_frame, text="Bad", variable=self.pose_type_var, value="bad", command=self.update_dataset_preview).pack()

        # Capture button
        self.capture_btn = ttk.Button(left_frame, text="Capture", command=self.capture_image)
        self.capture_btn.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(left_frame, length=200, maximum=self.max_captures)
        self.progress.pack(pady=10)

        # Dataset info
        self.dataset_info = ttk.Label(left_frame, text="")
        self.dataset_info.pack(pady=10)

        # Right frame for camera feed and dataset preview
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Camera feed
        self.camera_canvas = tk.Canvas(right_frame, width=640, height=480)
        self.camera_canvas.pack(pady=10)

        # Dataset preview
        preview_frame = ttk.Frame(right_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(preview_frame)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.preview_canvas.configure(yscrollcommand=scrollbar.set)
        self.preview_canvas.bind('<Configure>', lambda e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")))

        self.preview_frame = ttk.Frame(self.preview_canvas)
        self.preview_canvas.create_window((0, 0), window=self.preview_frame, anchor="nw")

    def populate_camera_list(self):
        camera_list = [f"Camera {i}" for i in self.cameras]
        self.camera_combo['values'] = camera_list
        if camera_list:
            self.camera_combo.current(0)

    def on_camera_change(self, event):
        if hasattr(self, 'cap'):
            self.cap.release()
        camera_index = self.cameras[self.camera_combo.current()]
        self.cap = cv2.VideoCapture(camera_index)

    def capture_image(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            camera_index = self.cameras[self.camera_combo.current()]
            self.cap = cv2.VideoCapture(camera_index)

        ret, frame = self.cap.read()
        if ret:
            pose_type = self.pose_type_var.get()
            existing_files = glob.glob(os.path.join(self.dataset_path, pose_type, f"{pose_type}_*.jpg"))
            next_index = len(existing_files)
            img_name = os.path.join(self.dataset_path, pose_type, f"{pose_type}_{next_index}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            self.capture_count += 1
            self.progress['value'] = self.capture_count

            if self.capture_count >= self.max_captures:
                self.capture_btn['state'] = 'disabled'
                print("Capture complete!")

            self.update_dataset_info()
            self.update_dataset_preview()

    def update_feed(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.camera_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.camera_canvas.image = photo
        self.master.after(10, self.update_feed)

    def update_dataset_info(self):
        good_count = len(glob.glob(os.path.join(self.dataset_path, 'good', '*.jpg')))
        bad_count = len(glob.glob(os.path.join(self.dataset_path, 'bad', '*.jpg')))
        info_text = f"Dataset Info:\nGood Poses: {good_count}\nBad Poses: {bad_count}"
        self.dataset_info.config(text=info_text)

    def update_dataset_preview(self):
        pose_type = self.pose_type_var.get()
        images = glob.glob(os.path.join(self.dataset_path, pose_type, '*.jpg'))
        images.sort(key=os.path.getmtime, reverse=True)
        
        # Clear previous preview
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        # Create a grid of images
        row = 0
        col = 0
        for img_path in images:
            img = Image.open(img_path)
            img.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(self.preview_frame, image=photo)
            label.image = photo
            label.grid(row=row, column=col, padx=5, pady=5)
            col += 1
            if col == 5:  # 5 images per row
                col = 0
                row += 1

        self.preview_canvas.update_idletasks()
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))

    def run(self):
        create_directories(self.dataset_path)
        self.update_feed()
        self.master.mainloop()

        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureCaptureApp(root)
    app.run()