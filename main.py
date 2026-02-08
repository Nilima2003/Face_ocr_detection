import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import os
import time
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import csv
import re
import sys
import shutil

# HELPER FUNCTIONS
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# OCR & FACE RECOGNITION CLASSES
class DoctrOCR:
    def __init__(self):
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            self.DocumentFile = DocumentFile
            self.model = ocr_predictor(pretrained=True)
            self.available = True
            print("OCR Ready")
        except Exception as e:
            print("OCR unavailable:", e)
            self.available = False

    def extract_numbers(self, img_path):
        if not self.available: return []
        doc = self.DocumentFile.from_images(img_path)
        result = self.model(doc)
        text = result.render()
        return re.findall(r"\b\d{4,9}\b", text)

class FaceRecognizer:
    def __init__(self, tolerance=0.6):
        try:
            import face_recognition
            self.face_recognition = face_recognition
            self.tolerance = tolerance
            self.available = True
            print("Face recognition ready")
        except Exception as e:
            print("Face recognition unavailable:", e)
            self.available = False

    def extract_face_encoding(self, image_path):
        if not self.available: return None
        try:
            image = self.face_recognition.load_image_file(image_path)
            encodings = self.face_recognition.face_encodings(image)
            return encodings[0] if encodings else None
        except Exception as e:
            print(f"Error extracting face from {image_path}: {e}")
            return None

    def compare_faces(self, encoding1, encoding2):
        if not self.available or encoding1 is None or encoding2 is None:
            return False
        try:
            results = self.face_recognition.compare_faces([encoding1], encoding2, tolerance=self.tolerance)
            return results[0] if results else False
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return False
# CONFIG & STYLING
SNAPSHOT_DIR = os.path.join(os.getcwd(), "snapshots")
CAMERA_CSV = resource_path("cameras.csv")
BOX_W, BOX_H = 150, 110 
MAX_BOXES = 5 
MOTION_THRESHOLD = 500
SNAP_COOLDOWN = 2
FACE_TOLERANCE = 0.6

COLOR_DARK_BLUE = "#335994"
COLOR_BG_WHITE = "#ffffff"
COLOR_GRID_GREEN = "#92c95d"

# CAMERA THREAD (Integrated)
class CameraWorker(threading.Thread):
    def __init__(self, cam_name, rtsp_url, ui_queue):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.rtsp_url = rtsp_url
        self.ui_queue = ui_queue
        self.running = True
        self.last_snap = 0
        self.ocr = DoctrOCR()
        self.face_recognizer = FaceRecognizer(tolerance=FACE_TOLERANCE)
        
        self.out_dir = os.path.join(SNAPSHOT_DIR, cam_name)
        self.face_alert_dir = os.path.join(self.out_dir, "FACE_ALERTS")
        os.makedirs(self.face_alert_dir, exist_ok=True)
        self.face_encoding_cache = {}

    def stop(self): self.running = False

    def get_all_existing_snapshots(self, exclude_path=None):
        all_snapshots = []
        for root, _, files in os.walk(self.out_dir):
            if "FACE_ALERTS" in root:
                continue
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    path = os.path.abspath(os.path.join(root, file))
                    if exclude_path:
                        if os.path.abspath(path)==os.path.abspath(exclude_path):
                            continue
                    all_snapshots.append(path)
        return all_snapshots

    def get_face_encoding_cached(self, path):
        if path not in self.face_encoding_cache:
            self.face_encoding_cache[path] = self.face_recognizer.extract_face_encoding(path)
        return self.face_encoding_cache[path]

    def process_face_logic(self, current_path, current_num, ts):
        if not self.face_recognizer.available: return None
        curr_enc = self.get_face_encoding_cached(current_path)
        if curr_enc is None: return None

        existing = self.get_all_existing_snapshots(exclude_path=current_path)
        alert_type = None

        # Rule 1: Same Face + Different Number
        for snap_path in existing:
            snap_num = os.path.basename(os.path.dirname(snap_path))
            snap_enc = self.get_face_encoding_cached(snap_path)
            if snap_enc is not None and self.face_recognizer.compare_faces(snap_enc, curr_enc):
                if snap_num != current_num:
                    alert_type = "SAME_FACE_DIFF_NUMBER"
                    break

        # Rule 2: Diff Face + Same Number
        if not alert_type:
            num_dir = os.path.join(self.out_dir, current_num)
            others = [os.path.join(num_dir, f) for f in os.listdir(num_dir) if f.endswith('.jpg') and not os.path.samefile(os.path.join(num_dir, f), current_path)]
            if others:
                is_new_face = True
                for other in others:
                    other_enc = self.get_face_encoding_cached(other)
                    if other_enc is not None and self.face_recognizer.compare_faces(other_enc, curr_enc):
                        is_new_face = False
                        break
                if is_new_face: alert_type = "DIFF_FACE_SAME_NUMBER"

        if alert_type:
            alert_dir = os.path.join(self.out_dir, current_num, "alert")
            os.makedirs(alert_dir, exist_ok=True)
            alert_fn = f"{ts}_{self.cam_name}_{current_num}_{alert_type}.jpg"
            shutil.copy(current_path, os.path.join(alert_dir, alert_fn))
            shutil.copy(current_path, os.path.join(self.face_alert_dir, alert_fn))
            return alert_type, os.path.join(alert_dir, alert_fn)
        return None

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened(): return
        ret, prev = cap.read()
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1); continue
            
            gray1, gray2 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if any(cv2.contourArea(c) > MOTION_THRESHOLD for c in cnts) and time.time() - self.last_snap > SNAP_COOLDOWN:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                tmp_path = os.path.join(self.out_dir, f"tmp_{ts}.jpg")
                cv2.imwrite(tmp_path, frame)
                numbers = self.ocr.extract_numbers(tmp_path)
                
                if numbers:
                    num = numbers[0]
                    num_path = os.path.join(self.out_dir, num)
                    os.makedirs(num_path, exist_ok=True)
                    final_path = os.path.join(num_path, f"{ts}_{self.cam_name}_{num}.jpg")
                    shutil.move(tmp_path, final_path)
                    
                    alert_info = self.process_face_logic(final_path, num, ts)
                    self.ui_queue.put(("snapshot", self.cam_name, final_path, f"{num} {'(!)' if alert_info else ''}"))
                    self.ui_queue.put(("alert_update",))
                    self.last_snap = time.time()
                elif os.path.exists(tmp_path): os.remove(tmp_path)
            prev = frame
            time.sleep(0.05)
        cap.release()

# =========================
# MAIN GUI
# =========================
class RTSPGui:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Face Monitoring System")
        self.root.configure(bg=COLOR_BG_WHITE)
        self.ui_queue = queue.Queue()
        self.workers, self.snap_boxes, self.snap_labels, self.images = {}, {}, {}, {}
        self.plus_buttons = {}
        self.gallery_img_refs = []
        self.alert_summary_labels = {}  # Store alert summary labels for each camera
        self.status_labels = {}  # Store camera status indicators

        self.build_ui()
        self.load_cameras_from_csv()
        self.root.after(200, self.process_queue)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Clean exit handler"""
        self.stop_all()
        self.root.destroy()

    def build_ui(self):
        header = tk.Frame(self.root, bg=COLOR_DARK_BLUE)
        header.pack(fill="x", padx=10, pady=5)
        tk.Label(header, text="OCR FACE MONITORING SYSTEM", bg=COLOR_DARK_BLUE, fg="white", 
                 font=("Arial", 14, "bold")).pack(pady=8)

        controls = tk.Frame(self.root, bg=COLOR_BG_WHITE)
        controls.pack(fill="x", padx=10, pady=5)
        btn_s = {"bg": COLOR_DARK_BLUE, "fg": "white", "font": ("Arial", 10), "width": 12, "bd": 0, "cursor": "hand2"}
        
        tk.Button(controls, text="Start", command=self.start_all, **btn_s).pack(side="left", padx=5)
        tk.Button(controls, text="Stop", command=self.stop_all, **btn_s).pack(side="left", padx=5)
        tk.Button(controls, text="View Log", command=self.show_logs, **btn_s).pack(side="left", padx=5)

        self.alert_btn = tk.Button(controls, text="Total Alert Summary (0)", command=self.show_camera_wise_alerts, 
                                   bg=COLOR_DARK_BLUE, fg="white", font=("Arial", 10), width=20, bd=0, cursor="hand2")
        self.alert_btn.pack(side="right", padx=5)

        outer = tk.Frame(self.root, bg="white", highlightbackground=COLOR_DARK_BLUE, highlightthickness=1)
        outer.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.stats_frame = tk.Frame(outer, bg=COLOR_DARK_BLUE)
        self.stats_frame.pack(anchor="nw", padx=10, pady=10)
        self.stats_label = tk.Label(self.stats_frame, text="Total Camera = 0", 
                                    bg=COLOR_DARK_BLUE, fg="white", font=("Arial", 10, "bold"),
                                    padx=10, pady=5)
        self.stats_label.pack()

        self.canvas = tk.Canvas(outer, bg="white", highlightthickness=0)
        self.scroll = ttk.Scrollbar(outer, command=self.canvas.yview)
        self.container = tk.Frame(self.canvas, bg="white")
        self.container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True, padx=10)
        self.scroll.pack(side="right", fill="y")

    def show_logs(self):
        log_win = tk.Toplevel(self.root)
        log_win.title("Detection Report & Logs")
        log_win.geometry("950x600")
        log_win.configure(bg=COLOR_BG_WHITE)

        tk.Label(log_win, text="DETECTION LOG REPORT", bg=COLOR_DARK_BLUE, fg="white", 
                 font=("Arial", 12, "bold"), pady=10).pack(fill="x")

        top_bar = tk.Frame(log_win, bg=COLOR_BG_WHITE)
        top_bar.pack(fill="x", padx=20, pady=10)
        
        frame = tk.Frame(log_win, bg="white")
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        columns = ("camera", "number", "datetime", "alert")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        tree.heading("camera", text="Camera Name")
        tree.heading("number", text="Detected Number")
        tree.heading("datetime", text="Date & Time")
        tree.heading("alert", text="Alert Status")

        tree.column("camera", width=200, anchor="w")
        tree.column("number", width=150, anchor="center")
        tree.column("datetime", width=200, anchor="center")
        tree.column("alert", width=120, anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        log_data = []
        if os.path.exists(SNAPSHOT_DIR):
            for cam in os.listdir(SNAPSHOT_DIR):
                cam_path = os.path.join(SNAPSHOT_DIR, cam)
                if not os.path.isdir(cam_path): continue
                for num in os.listdir(cam_path):
                    num_path = os.path.join(cam_path, num)
                    if os.path.isdir(num_path) and num.isdigit():
                        alert_dir = os.path.join(num_path, "alert")
                        for file in os.listdir(num_path):
                            if file.lower().endswith(('.jpg', '.png')):
                                is_alert = "Yes" if os.path.exists(os.path.join(alert_dir, file)) else "No"
                                parts = file.split('_')
                                if len(parts) >= 2:
                                    dt = f"{parts[0]} {parts[1].replace('-', ':')}"
                                    log_data.append((cam, num, dt, is_alert))

        log_data.sort(key=lambda x: x[2], reverse=True)
        tree.tag_configure("is_alert", foreground="red")

        for row in log_data:
            if row[3] == "Yes":
                tree.insert("", "end", values=row, tags=("is_alert",))
            else:
                tree.insert("", "end", values=row)

        def download_csv():
            if not log_data:
                messagebox.showwarning("Download", "No report data to export.")
                return
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV files", "*.csv")],
                                                     initialfile=f"OCR_Report_{datetime.now().strftime('%Y%m%d')}")
            if file_path:
                try:
                    with open(file_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Camera Name", "Detected Number", "Date & Time", "Alert Status"])
                        writer.writerows(log_data)
                    messagebox.showinfo("Success", f"CSV Report saved to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save CSV: {e}")

        tk.Button(top_bar, text="Download Report CSV", command=download_csv, bg=COLOR_GRID_GREEN, 
                  fg="white", font=("Arial", 10, "bold"), bd=0, padx=15, pady=5, cursor="hand2").pack(side="right")

    def view_image(self, path, title):
        view_win = tk.Toplevel(self.root)
        view_win.title(f"Viewing: {title}")
        try:
            img = Image.open(path)
            img.thumbnail((1100, 800))
            tkimg = ImageTk.PhotoImage(img)
            lbl = tk.Label(view_win, image=tkimg, bg="black")
            lbl.image = tkimg 
            lbl.pack(padx=10, pady=10)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def show_camera_wise_alerts(self):
        alert_win = tk.Toplevel(self.root)
        alert_win.title("Total Alert Summary")
        alert_win.geometry("500x500")
        alert_win.configure(bg=COLOR_BG_WHITE)

        tk.Label(alert_win, text="CAMERA WISE ALERT SUMMARY", bg=COLOR_DARK_BLUE, fg="white", 
                 font=("Arial", 12, "bold"), pady=10).pack(fill="x")

        container = tk.Frame(alert_win, bg=COLOR_BG_WHITE)
        container.pack(fill="both", expand=True, padx=15, pady=15)
        
        canv = tk.Canvas(container, bg=COLOR_BG_WHITE, highlightthickness=0)
        scr = ttk.Scrollbar(container, orient="vertical", command=canv.yview)
        scroll_frame = tk.Frame(canv, bg=COLOR_BG_WHITE)

        scroll_frame.bind("<Configure>", lambda e: canv.configure(scrollregion=canv.bbox("all")))
        canv.create_window((0, 0), window=scroll_frame, anchor="nw")
        canv.configure(yscrollcommand=scr.set)
        canv.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")

        for cam in sorted(self.snap_boxes.keys()):
            cam_path = os.path.join(SNAPSHOT_DIR, cam)
            alert_count = 0
            detected_nums = []

            if os.path.exists(cam_path):
                for item in os.listdir(cam_path):
                    if item.isdigit():
                        alert_folder = os.path.join(cam_path, item, "alert")
                        if os.path.exists(alert_folder):
                            files = [f for f in os.listdir(alert_folder) if f.lower().endswith(('.jpg', '.png'))]
                            if files:
                                alert_count += len(files)
                                detected_nums.append(item)
            
            unique_nums_str = ", ".join(sorted(list(set(detected_nums))))
            cam_row = tk.Frame(scroll_frame, bg=COLOR_BG_WHITE, pady=10)
            cam_row.pack(fill="x")
            header_row = tk.Frame(cam_row, bg=COLOR_BG_WHITE)
            header_row.pack(fill="x")
            tk.Label(header_row, text=f"• {cam}", bg=COLOR_BG_WHITE, font=("Arial", 10, "bold")).pack(side="left")
            tk.Label(header_row, text=f"{alert_count} Alerts", bg=COLOR_BG_WHITE, fg="red", font=("Arial", 10)).pack(side="right")
            
            if alert_count > 0:
                tk.Label(cam_row, text=f"   Detected Numbers: {{{unique_nums_str}}}", 
                         bg=COLOR_BG_WHITE, fg="#444444", font=("Arial", 9), 
                         wraplength=400, justify="left").pack(fill="x", padx=20)
            else:
                tk.Label(cam_row, text="   No alerts detected", bg=COLOR_BG_WHITE, 
                         fg="#999999", font=("Arial", 9, "italic")).pack(fill="x", padx=20)
            
            tk.Frame(scroll_frame, height=1, bg="#eeeeee").pack(fill="x", pady=2)

        tk.Button(alert_win, text="Done", command=alert_win.destroy, bg=COLOR_DARK_BLUE, fg="white", width=12).pack(pady=15)

    def add_camera_row(self, cam):
        row = tk.Frame(self.container, bg="white", pady=15)
        row.pack(fill="x")
        
        # Camera status indicator
        status_frame = tk.Frame(row, bg="white", width=20)
        status_frame.pack(side="left", padx=5)
        status_frame.pack_propagate(False)
        
        self.status_labels[cam] = tk.Label(status_frame, text="●", 
                                           font=("Arial", 16), fg="gray", bg="white")
        self.status_labels[cam].pack(expand=True)
        
        # Camera Name Sidebar
        tk.Label(row, text=cam, bg="white", font=("Arial", 11, "bold"), width=12, anchor="w").pack(side="left", padx=5)
        
        self.images[cam] = []
        boxes, labels = [], []
        grid_frame = tk.Frame(row, bg="white")
        grid_frame.pack(side="left")
        
        # 6 Boxes for images + 1 for plus button
        for i in range(MAX_BOXES):
            col = tk.Frame(grid_frame, bg="white")
            col.pack(side="left", padx=5)
            
            if i == MAX_BOXES - 1:  # Last position for plus button
                # Plus button container (same size as image boxes)
                plus_container = tk.Frame(col, width=BOX_W, height=BOX_H, bg="#f0f0f0", 
                                         highlightthickness=1, highlightbackground="#cccccc")
                plus_container.pack_propagate(False)
                plus_container.pack()
                
                # Plus button label (acts as button)
                plus_lbl = tk.Label(plus_container, text="+", bg="#f0f0f0", 
                                   font=("Arial", 36, "bold"), fg=COLOR_DARK_BLUE,
                                   cursor="hand2")
                plus_lbl.pack(fill="both", expand=True)
                
                # Bind click to open gallery
                plus_lbl.bind("<Button-1>", lambda e, c=cam: self.open_gallery(c))
                
                # Empty text label below plus button
                txt_lbl = tk.Label(col, text="Gallery", bg="white", fg="#555555", 
                                  font=("Arial", 7), wraplength=BOX_W, justify="center")
                txt_lbl.pack(pady=2)
                
                boxes.append(plus_lbl)  # Store plus label in boxes list
                labels.append(txt_lbl)
                
            else:
                # Fixed size container for the image
                img_container = tk.Frame(col, width=BOX_W, height=BOX_H, bg="#f0f0f0", 
                                        highlightthickness=1, highlightbackground="#cccccc")
                img_container.pack_propagate(False)
                img_container.pack()
                
                img_lbl = tk.Label(img_container, bg="#f0f0f0")
                img_lbl.pack(fill="both", expand=True)
                
                # Bind double-click to open gallery, single-click to view image
                img_lbl.bind("<Double-Button-1>", lambda e, c=cam: self.open_gallery(c))
                
                # Name/Timestamp label ALWAYS below the box
                txt_lbl = tk.Label(col, text="-", bg="white", fg="#555555", 
                                  font=("Arial", 7), wraplength=BOX_W, justify="center")
                txt_lbl.pack(pady=2)
                
                boxes.append(img_lbl)
                labels.append(txt_lbl)
        
        self.snap_boxes[cam], self.snap_labels[cam] = boxes, labels
        
        # ============ ADD ALERT SUMMARY FRAME ON RIGHT SIDE ============
        # Create frame for alert summary (right side of the row)
        alert_frame = tk.Frame(row, bg="white", relief="ridge", 
                              borderwidth=1, highlightbackground=COLOR_DARK_BLUE)
        alert_frame.pack(side="right", padx=20, fill="y", expand=False)
        
        # Alert summary title
        alert_title = tk.Label(alert_frame, text="Alert Summary", 
                              bg=COLOR_DARK_BLUE, fg="white", 
                              font=("Arial", 10, "bold"), pady=5, width=20)
        alert_title.pack(fill="x")
        
        # Create a frame to hold alert stats
        stats_container = tk.Frame(alert_frame, bg="white", pady=10)
        stats_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Total Alerts label with dynamic update
        total_alert_label = tk.Label(stats_container, text="Total Alert = 0", 
                                    bg="white", font=("Arial", 10), fg="#333333")
        total_alert_label.pack(anchor="w", pady=2)
        
        # Detected Numbers label with dynamic update
        numbers_label = tk.Label(stats_container, text="Numbers: -", 
                                bg="white", font=("Arial", 9), fg="#555555",
                                wraplength=180, justify="left", anchor="w")
        numbers_label.pack(anchor="w", pady=2)
        
        # Store references for updating
        self.alert_summary_labels[cam] = {
            'total': total_alert_label,
            'numbers': numbers_label,
            'frame': alert_frame
        }
        
        # Initialize alert stats
        self.update_camera_alert_summary(cam)

    def process_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                if msg[0] == "snapshot": 
                    self.update_ui_slots(msg[1], msg[2], msg[3])
                    # Update specific camera's alert summary
                    self.update_camera_alert_summary(msg[1])
                    self.update_total_stats()
                elif msg[0] == "alert_update": 
                    self.update_total_stats()
                    # Update all camera summaries
                    self.update_camera_alert_summary()
        except queue.Empty: 
            pass
        self.root.after(200, self.process_queue)

    def update_total_stats(self):
        cam_count = len(self.snap_boxes)
        self.stats_label.config(text=f"Total Camera = {cam_count}")

        # Calculate Total Alerts by checking directory folders
        total_alerts = 0
        if os.path.exists(SNAPSHOT_DIR):
            for root_dir, dirs, files in os.walk(SNAPSHOT_DIR):
                if "alert" in root_dir.lower():
                    images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    total_alerts += len(images)
        
        self.alert_btn.config(text=f"Total Alert Summary ({total_alerts})")

    def update_ui_slots(self, cam, path, filename):
        try:
            # Resize image to fill the exact box dimensions (BOX_W, BOX_H)
            img = Image.open(path).resize((BOX_W, BOX_H), Image.Resampling.LANCZOS)
            tkimg = ImageTk.PhotoImage(img)
            
            # Add new image to the beginning (most recent)
            self.images[cam].insert(0, (tkimg, filename, path))
            
            # Keep only MAX_BOXES-1 images (last spot is for plus button)
            self.images[cam] = self.images[cam][:(MAX_BOXES-1)]
            
            # Update all image slots, but skip the plus button position
            for i in range(MAX_BOXES - 1):  # Only process first 5 positions (0-4)
                if i < len(self.images[cam]):  # If we have an image for this slot
                    im, name, full_path = self.images[cam][i]
                    slot = self.snap_boxes[cam][i]
                    
                    # Configure the image
                    slot.configure(image=im)
                    slot.image = im
                    
                    # Update text label below
                    self.snap_labels[cam][i].configure(text=name)
                    
                    # Bind events
                    slot.unbind("<Button-1>")
                    slot.unbind("<Double-Button-1>")
                    
                    # Single click to view image
                    slot.bind("<Button-1>", lambda e, p=full_path, f=name, c=cam: 
                             self.view_image(p, f))
                    
                    # Double click to open gallery
                    slot.bind("<Double-Button-1>", lambda e, c=cam: self.open_gallery(c))
                    slot.config(cursor="hand2")
                else:
                    # Clear empty slots
                    slot = self.snap_boxes[cam][i]
                    slot.configure(image='')
                    self.snap_labels[cam][i].configure(text="-")
                    
        except Exception as e:
            print(f"Error updating UI slots: {e}")

    def update_camera_alert_summary(self, cam_name=None):
        """Update alert summary for specific camera or all cameras"""
        cameras_to_update = [cam_name] if cam_name else list(self.alert_summary_labels.keys())
        
        for cam in cameras_to_update:
            if cam not in self.alert_summary_labels:
                continue
                
            cam_path = os.path.join(SNAPSHOT_DIR, cam)
            alert_count = 0
            detected_numbers = []
            
            if os.path.exists(cam_path):
                # Walk through the camera directory
                for root_dir, dirs, files in os.walk(cam_path):
                    # Check if this is an alert directory
                    if "alert" in root_dir.lower():
                        # Count image files in alert folders
                        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        alert_count += len(images)
                        
                        # Extract number from directory structure
                        # Path structure: snapshots/cam_name/number/alert/
                        path_parts = root_dir.split(os.sep)
                        for i, part in enumerate(path_parts):
                            if part.isdigit() and i < len(path_parts) - 1:
                                if part not in detected_numbers:
                                    detected_numbers.append(part)
                                break
            
            # Update the labels
            total_label = self.alert_summary_labels[cam]['total']
            numbers_label = self.alert_summary_labels[cam]['numbers']
            
            # Update total alerts
            total_label.config(text=f"Total Alert = {alert_count}")
            
            # Update detected numbers
            if detected_numbers:
                # Format numbers for display (show first 3 unique numbers)
                unique_nums = sorted(list(set(detected_numbers)))
                if len(unique_nums) > 3:
                    nums_text = f"Numbers: {', '.join(unique_nums[:3])}... ({len(unique_nums)} total)"
                else:
                    nums_text = f"Numbers: {', '.join(unique_nums)}"
                
                # Highlight if there are alerts
                if alert_count > 0:
                    total_label.config(fg="red", font=("Arial", 10, "bold"))
                    numbers_label.config(fg="#cc0000")
                else:
                    total_label.config(fg="#333333", font=("Arial", 10))
                    numbers_label.config(fg="#555555")
                
                numbers_label.config(text=nums_text)
            else:
                total_label.config(fg="#333333", font=("Arial", 10))
                numbers_label.config(text="Numbers: -", fg="#555555")
            
            # Update frame border color if there are alerts
            alert_frame = self.alert_summary_labels[cam]['frame']
            if alert_count > 0:
                alert_frame.config(highlightbackground="red", highlightthickness=2)
            else:
                alert_frame.config(highlightbackground=COLOR_DARK_BLUE, highlightthickness=1)
    
    #def refresh_all_alert_summaries(self):
        #"""Force refresh of all camera alert summaries"""
        #self.update_camera_alert_summary()
        #self.update_total_stats()

    def load_cameras_from_csv(self):
        if not os.path.exists(CAMERA_CSV): return
        with open(CAMERA_CSV, newline="") as f:
            for cam, rtsp in csv.reader(f):
                self.add_camera_row(cam.strip())
                self.workers[f"{cam.strip()}_url"] = rtsp.strip()
        self.update_total_stats()

    def start_all(self):
        for cam in self.snap_boxes.keys():
            url_key = f"{cam}_url"
            if url_key in self.workers and not isinstance(self.workers.get(cam), CameraWorker):
                w = CameraWorker(cam, self.workers[url_key], self.ui_queue)
                self.workers[cam] = w
                w.start()
                # Update status indicator
                if cam in self.status_labels:
                    self.status_labels[cam].config(fg="green")

    def stop_all(self):
        for cam, worker in list(self.workers.items()):
            if isinstance(worker, CameraWorker): 
                worker.stop()
                del self.workers[cam]
                # Update status indicator
                if cam in self.status_labels:
                    self.status_labels[cam].config(fg="gray")
        
        # Reset alert summaries to zero
        self.update_camera_alert_summary()

    def open_gallery(self, cam):
        win = tk.Toplevel(self.root)
        win.title(f"Gallery - {cam}")
        win.geometry("1100x800")
        win.configure(bg=COLOR_BG_WHITE)
        folder = os.path.join(SNAPSHOT_DIR, cam)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: All Images
        all_tab = ttk.Frame(notebook)
        notebook.add(all_tab, text="All Images")
        
        # Tab 2: Alert Images
        alert_tab = ttk.Frame(notebook)
        notebook.add(alert_tab, text="Alert Images")
        
        # Tab 3: By Number
        number_tab = ttk.Frame(notebook)
        notebook.add(number_tab, text="By Number")
        
        # Load each tab
        self.load_all_images_tab(all_tab, folder)
        self.load_alert_images_tab(alert_tab, folder)
        self.load_number_view_tab(number_tab, folder)

    def load_all_images_tab(self, container, folder):
        """Load all images tab"""
        # Clear existing widgets
        for widget in container.winfo_children():
            widget.destroy()
        
        if not os.path.exists(folder):
            tk.Label(container, text="No snapshots folder found", 
                    bg=COLOR_BG_WHITE, fg="#666666", font=("Arial", 11)).pack(pady=50)
            return
        
        # Create scrollable canvas
        canvas = tk.Canvas(container, bg=COLOR_BG_WHITE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG_WHITE)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Collect all image files
        image_files = []
        for root_dir, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root_dir, file)
                    image_files.append(full_path)
        
        if not image_files:
            tk.Label(scrollable_frame, text="No images found", 
                    bg=COLOR_BG_WHITE, fg="#666666", font=("Arial", 11)).pack(pady=50)
            return
        
        # Sort by filename (most recent first)
        image_files.sort(key=lambda x: os.path.basename(x), reverse=True)
        
        # Display images in grid
        self.gallery_img_refs = []
        row_frame = None
        
        for i, img_path in enumerate(image_files):
            if i % 5 == 0:  # 5 images per row
                row_frame = tk.Frame(scrollable_frame, bg=COLOR_BG_WHITE)
                row_frame.pack(pady=10)
            
            try:
                # Load and resize image
                img = Image.open(img_path)
                img.thumbnail((200, 150), Image.Resampling.LANCZOS)
                tkimg = ImageTk.PhotoImage(img)
                self.gallery_img_refs.append(tkimg)
                
                # Create frame for image
                frame = tk.Frame(row_frame, bg="white", relief="ridge", borderwidth=1)
                frame.grid(row=0, column=i % 5, padx=10)
                
                # Image label
                img_label = tk.Label(frame, image=tkimg, bg="white", cursor="hand2")
                img_label.image = tkimg  # Keep reference
                img_label.pack(padx=5, pady=5)
                
                # Bind click event
                img_label.bind("<Button-1>", 
                             lambda e, p=img_path, f=os.path.basename(img_path): 
                             self.view_image(p, f))
                
                # Filename label
                filename = os.path.basename(img_path)
                tk.Label(frame, text=filename, bg="white", 
                        font=("Arial", 8), wraplength=190).pack(pady=5)
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

    def load_alert_images_tab(self, container, folder):
        """Load alert images tab"""
        # Clear existing widgets
        for widget in container.winfo_children():
            widget.destroy()
        
        if not os.path.exists(folder):
            tk.Label(container, text="No snapshots folder found", 
                    bg=COLOR_BG_WHITE, fg="#666666", font=("Arial", 11)).pack(pady=50)
            return
        
        # Create scrollable canvas
        canvas = tk.Canvas(container, bg=COLOR_BG_WHITE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG_WHITE)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Collect alert image files
        alert_files = []
        for root_dir, dirs, files in os.walk(folder):
            if "alert" in root_dir.lower():
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root_dir, file)
                        alert_files.append(full_path)
        
        if not alert_files:
            tk.Label(scrollable_frame, text="No alert images found", 
                    bg=COLOR_BG_WHITE, fg="#666666", font=("Arial", 11)).pack(pady=50)
            return
        
        # Sort by filename (most recent first)
        alert_files.sort(key=lambda x: os.path.basename(x), reverse=True)
        
        # Display images in grid
        self.gallery_img_refs = []
        row_frame = None
        
        for i, img_path in enumerate(alert_files):
            if i % 5 == 0:  # 5 images per row
                row_frame = tk.Frame(scrollable_frame, bg=COLOR_BG_WHITE)
                row_frame.pack(pady=10)
            
            try:
                # Load and resize image
                img = Image.open(img_path)
                img.thumbnail((200, 150), Image.Resampling.LANCZOS)
                tkimg = ImageTk.PhotoImage(img)
                self.gallery_img_refs.append(tkimg)
                
                # Create frame for image (red border for alerts)
                frame = tk.Frame(row_frame, bg="white", relief="ridge", 
                                borderwidth=2, highlightbackground="red")
                frame.grid(row=0, column=i % 5, padx=10)
                
                # Image label
                img_label = tk.Label(frame, image=tkimg, bg="white", cursor="hand2")
                img_label.image = tkimg  # Keep reference
                img_label.pack(padx=5, pady=5)
                
                # Bind click event
                img_label.bind("<Button-1>", 
                             lambda e, p=img_path, f=os.path.basename(img_path): 
                             self.view_image(p, f))
                
                # Filename label
                filename = os.path.basename(img_path)
                tk.Label(frame, text=filename, bg="white", 
                        font=("Arial", 8), fg="red", wraplength=190).pack(pady=5)
                
            except Exception as e:
                print(f"Error loading alert image {img_path}: {e}")
                continue

    def load_number_view_tab(self, container, folder):
        """Load by number tab with scrollbar for all numbers"""
        # Clear existing widgets
        for widget in container.winfo_children():
            widget.destroy()
        
        if not os.path.exists(folder):
            tk.Label(container, text="No snapshots folder found", 
                    bg=COLOR_BG_WHITE, fg="#666666", font=("Arial", 11)).pack(pady=50)
            return
        
        # Create main frame with scrollbar
        main_frame = tk.Frame(container, bg=COLOR_BG_WHITE)
        main_frame.pack(fill="both", expand=True)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(main_frame, bg=COLOR_BG_WHITE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG_WHITE)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Get all number directories
        number_dirs = []
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path):
                # Check if it's a number directory
                if item.isdigit():
                    number_dirs.append(item)
        
        if not number_dirs:
            tk.Label(scrollable_frame, text="No number directories found", 
                    bg=COLOR_BG_WHITE, fg="#666666", font=("Arial", 11)).pack(pady=50)
            return
        
        # Sort numbers
        number_dirs.sort()
        self.gallery_img_refs = []
        
        # Create a frame to hold all number sections
        content_frame = tk.Frame(scrollable_frame, bg=COLOR_BG_WHITE)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        for number in number_dirs:
            # Create a frame for this number section
            num_section_frame = tk.Frame(content_frame, bg=COLOR_BG_WHITE, relief="ridge", borderwidth=1)
            num_section_frame.pack(fill="x", padx=5, pady=10, expand=True)
            
            # Number header with count
            num_path = os.path.join(folder, number)
            image_files = []
            
            if os.path.exists(num_path):
                # Get regular images
                for file in os.listdir(num_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and file != "alert":
                        image_files.append(os.path.join(num_path, file))
                
                # Get alert images
                alert_path = os.path.join(num_path, "alert")
                if os.path.exists(alert_path):
                    for file in os.listdir(alert_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_files.append(os.path.join(alert_path, file))
            
            # Count alerts
            alert_count = 0
            if os.path.exists(os.path.join(num_path, "alert")):
                alert_dir = os.path.join(num_path, "alert")
                if os.path.exists(alert_dir):
                    alert_count = len([f for f in os.listdir(alert_dir) if f.lower().endswith(('.jpg', '.png'))])
            
            # Header frame
            header_frame = tk.Frame(num_section_frame, bg=COLOR_BG_WHITE)
            header_frame.pack(fill="x", padx=10, pady=5)
            
            # Number label
            tk.Label(header_frame, text=f"Number: {number}", 
                    bg=COLOR_BG_WHITE, font=("Arial", 11, "bold")).pack(side="left")
            
            # Count labels
            count_frame = tk.Frame(header_frame, bg=COLOR_BG_WHITE)
            count_frame.pack(side="right")
            
            total_count = len(image_files)
            tk.Label(count_frame, text=f"Total: {total_count}", 
                    bg=COLOR_BG_WHITE, font=("Arial", 9)).pack(side="left", padx=5)
            
            if alert_count > 0:
                tk.Label(count_frame, text=f"Alerts: {alert_count}", 
                        bg=COLOR_BG_WHITE, fg="red", font=("Arial", 9, "bold")).pack(side="left", padx=5)
            
            # Separator line
            tk.Frame(num_section_frame, height=1, bg="#dddddd").pack(fill="x", padx=10, pady=5)
            
            if not image_files:
                tk.Label(num_section_frame, text=f"No images for number {number}", 
                        bg=COLOR_BG_WHITE, fg="#999999", font=("Arial", 9, "italic")).pack(pady=20)
                continue
            
            # Sort by filename (most recent first)
            image_files.sort(key=lambda x: os.path.basename(x), reverse=True)
            
            # Create canvas for horizontal scrolling of images
            images_canvas = tk.Canvas(num_section_frame, bg=COLOR_BG_WHITE, height=130, 
                                     highlightthickness=0)
            images_scrollbar = ttk.Scrollbar(num_section_frame, orient="horizontal", 
                                           command=images_canvas.xview)
            images_frame = tk.Frame(images_canvas, bg=COLOR_BG_WHITE)
            
            images_frame.bind(
                "<Configure>",
                lambda e, c=images_canvas: c.configure(scrollregion=c.bbox("all"))
            )
            
            images_canvas.create_window((0, 0), window=images_frame, anchor="nw")
            images_canvas.configure(xscrollcommand=images_scrollbar.set)
            
            images_canvas.pack(fill="x", padx=10, pady=5)
            images_scrollbar.pack(fill="x", padx=10)
            
            # Display images in a horizontal row
            for i, img_path in enumerate(image_files):
                try:
                    # Load and resize image
                    img = Image.open(img_path)
                    img.thumbnail((100, 75), Image.Resampling.LANCZOS)
                    tkimg = ImageTk.PhotoImage(img)
                    self.gallery_img_refs.append(tkimg)
                    
                    # Check if it's an alert image
                    is_alert = "alert" in img_path.lower()
                    
                    # Create frame for image
                    if is_alert:
                        frame = tk.Frame(images_frame, bg="white", relief="ridge", 
                                        borderwidth=2, highlightbackground="red")
                    else:
                        frame = tk.Frame(images_frame, bg="white", relief="ridge", borderwidth=1)
                    
                    frame.pack(side="left", padx=5, pady=5)
                    
                    # Image label
                    img_label = tk.Label(frame, image=tkimg, bg="white", cursor="hand2")
                    img_label.image = tkimg  # Keep reference
                    img_label.pack(padx=2, pady=2)
                    
                    # Bind click event
                    img_label.bind("<Button-1>", 
                                 lambda e, p=img_path, f=os.path.basename(img_path): 
                                 self.view_image(p, f))
                    
                    # Filename label (shortened)
                    filename = os.path.basename(img_path)
                    # Extract just the timestamp part (first 19 chars: YYYY-MM-DD_HH-MM-SS)
                    if '_' in filename:
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            timestamp = f"{parts[0]}_{parts[1]}"
                            if len(timestamp) > 19:
                                timestamp = timestamp[:19]
                        else:
                            timestamp = filename[:15] + "..."
                    else:
                        timestamp = filename[:15] + "..."
                    
                    if is_alert:
                        tk.Label(frame, text=timestamp, bg="white", 
                                font=("Arial", 7), fg="red", wraplength=90).pack()
                    else:
                        tk.Label(frame, text=timestamp, bg="white", 
                                font=("Arial", 7), wraplength=90).pack()
                    
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue


if __name__ == "__main__":
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    root = tk.Tk()
    root.geometry("1600x900") 
    RTSPGui(root)
    root.mainloop()