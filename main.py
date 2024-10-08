import cv2
import os
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from data_loader import load_datasets
from model import PostureNet
from train import train_model
from posture_detector import detect_posture
import threading
import queue
import sys
import io
import subprocess

class StreamToQueue(io.TextIOBase):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(("progress", text))

class PostureDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Posture Detection")
        
        # Remove the fixed geometry
        # self.master.geometry("1200x800")
        
        # Maximize the window
        self.master.state('zoomed')  # For Windows
        # self.master.attributes('-zoomed', True)  # For Linux
        # self.master.state('zoomed')  # For macOS
        
        load_dotenv()
        self.dataset_path = os.getenv('DATASET_PATH')
        self.model_path = os.getenv('MODEL_PATH')

        self.setup_ui()
        self.cameras = self.list_available_cameras()
        self.populate_camera_list()

        self.training_thread = None
        self.training_queue = queue.Queue()

    def setup_ui(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for controls
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Dataset selection
        ttk.Label(left_frame, text="Dataset Path:").pack(pady=5)
        self.dataset_entry = ttk.Entry(left_frame)
        self.dataset_entry.pack(fill=tk.X, padx=5, pady=5)
        self.dataset_entry.insert(0, self.dataset_path)
        ttk.Button(left_frame, text="Browse", command=self.browse_dataset).pack(pady=5)

        # Model selection
        ttk.Label(left_frame, text="Model Path:").pack(pady=5)
        self.model_entry = ttk.Entry(left_frame)
        self.model_entry.pack(fill=tk.X, padx=5, pady=5)
        self.model_entry.insert(0, self.model_path)
        ttk.Button(left_frame, text="Browse", command=self.browse_model).pack(pady=5)

        # ResNet option
        self.use_resnet_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Use ResNet50", variable=self.use_resnet_var).pack(pady=5)

        # Train button
        self.train_button = ttk.Button(left_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        # Camera selection
        ttk.Label(left_frame, text="Select Camera:").pack(pady=5)
        self.camera_combo = ttk.Combobox(left_frame)
        self.camera_combo.pack(pady=5)

        # Start/Stop detection
        self.detect_button = ttk.Button(left_frame, text="Start Detection", command=self.toggle_detection)
        self.detect_button.pack(pady=10)

        # Add a text area for displaying training progress
        self.progress_text = scrolledtext.ScrolledText(left_frame, height=20, width=50)
        self.progress_text.pack(pady=10, fill=tk.BOTH, expand=True)

        # Right frame for camera feed and take pose button
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.camera_canvas = tk.Canvas(right_frame, width=640, height=480)
        self.camera_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Add Take Pose button
        self.take_pose_button = ttk.Button(right_frame, text="Take Pose", command=self.run_take_pose)
        self.take_pose_button.pack(pady=10)

    def browse_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, path)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)

    def list_available_cameras(self):
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

    def populate_camera_list(self):
        camera_list = [f"Camera {i}" for i in self.cameras]
        self.camera_combo['values'] = camera_list
        if camera_list:
            self.camera_combo.current(0)

    def train_model(self):
        dataset_path = self.dataset_entry.get()
        model_path = self.model_entry.get()
        use_resnet = self.use_resnet_var.get()

        # Disable the train button
        self.train_button['state'] = 'disabled'

        # Clear the progress text
        self.progress_text.delete('1.0', tk.END)

        # Start the training in a separate thread
        self.training_thread = threading.Thread(target=self._train_model_thread, 
                                                args=(dataset_path, model_path, use_resnet))
        self.training_thread.start()

        # Start checking the queue for updates
        self.master.after(100, self._check_training_queue)

    def _train_model_thread(self, dataset_path, model_path, use_resnet):
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = StreamToQueue(self.training_queue)

        try:
            train_data, train_labels, val_data, val_labels = load_datasets(dataset_path)
            model = PostureNet(use_resnet=use_resnet)
            trained_model = train_model(model, train_data, train_labels, val_data, val_labels, None)
            trained_model.save(model_path)

            self.training_queue.put(("complete", f"Model saved to {model_path}"))
        except Exception as e:
            self.training_queue.put(("error", str(e)))
        finally:
            # Restore stdout
            sys.stdout = old_stdout

    def _check_training_queue(self):
        try:
            while True:  # Process all available messages
                message_type, message = self.training_queue.get_nowait()
                if message_type == "progress":
                    self.progress_text.insert(tk.END, message)
                    self.progress_text.see(tk.END)
                elif message_type == "complete":
                    self.progress_text.insert(tk.END, "\nTraining Complete!\n")
                    self.progress_text.see(tk.END)
                    messagebox.showinfo("Training Complete", message)
                    self.train_button['state'] = 'normal'
                elif message_type == "error":
                    self.progress_text.insert(tk.END, f"\nError: {message}\n")
                    self.progress_text.see(tk.END)
                    messagebox.showerror("Error", message)
                    self.train_button['state'] = 'normal'
        except queue.Empty:
            pass

        # If training is still running, check again after 100ms
        if self.training_thread and self.training_thread.is_alive():
            self.master.after(100, self._check_training_queue)
        else:
            self.train_button['state'] = 'normal'

    def toggle_detection(self):
        if self.detect_button['text'] == "Start Detection":
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        model_path = self.model_entry.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "No trained model found. Please train the model first.")
            return

        self.trained_model = PostureNet(use_resnet=self.use_resnet_var.get())
        self.trained_model.load_weights(model_path)

        camera_index = self.cameras[self.camera_combo.current()]
        self.cap = cv2.VideoCapture(camera_index)
        self.detect_button['text'] = "Stop Detection"
        self.update_detection()

    def stop_detection(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        self.detect_button['text'] = "Start Detection"
        self.camera_canvas.delete("all")

    def update_detection(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                posture = detect_posture(frame, self.trained_model)
                
                # Create a copy of the frame to draw on
                display_frame = frame.copy()
                
                # Set text color and border color based on posture
                if posture == "Bad":
                    text_color = (0, 0, 255)  # Red for BGR
                    border_color = (0, 0, 255)  # Red for BGR
                    # Add red border
                    display_frame = cv2.copyMakeBorder(display_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
                else:
                    text_color = (0, 255, 0)  # Green for BGR
                
                # Display the result on the frame
                cv2.putText(display_frame, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                self.camera_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.camera_canvas.image = photo

            self.master.after(10, self.update_detection)

    def run_take_pose(self):
        self.take_pose_button['state'] = 'disabled'
        self.progress_text.insert(tk.END, "Running take_pose.py...\n")
        self.progress_text.see(tk.END)

        def run_script():
            try:
                result = subprocess.run(["python", "take_pose.py"], 
                                        check=True, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        text=True)
                self.training_queue.put(("progress", result.stdout))
                self.training_queue.put(("progress", result.stderr))
                self.training_queue.put(("complete", "take_pose.py completed successfully."))
            except subprocess.CalledProcessError as e:
                self.training_queue.put(("error", f"Error running take_pose.py: {e}"))
            finally:
                self.master.after(0, lambda: self.take_pose_button.config(state='normal'))

        thread = threading.Thread(target=run_script)
        thread.start()

        # Start checking the queue for updates
        self.master.after(100, self._check_training_queue)

    def run(self):
        self.master.mainloop()

        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureDetectionApp(root)
    app.run()