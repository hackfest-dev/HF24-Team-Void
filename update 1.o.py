import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from cvzone.PoseModule import PoseDetector

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Void BioTracker")
        self.geometry("600x200")
        self.is_paused = False

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.title_label = ttk.Label(self.main_frame, text="Void BioTracker", font=("Arial", 20, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.live_camera_frame = ttk.LabelFrame(self.main_frame, text="Live Camera Test")
        self.live_camera_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.live_camera_button = ttk.Button(self.live_camera_frame, text="Start Wireframing Camera", command=self.on_click1)
        self.live_camera_button.pack(padx=10, pady=10)

        self.prerecorded_video_frame = ttk.LabelFrame(self.main_frame, text="Prerecorded Video Test")
        self.prerecorded_video_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.video_name_label = ttk.Label(self.prerecorded_video_frame, text="Video Path:")
        self.video_name_label.grid(row=0, column=0, padx=5, pady=5)

        self.video_name_entry = ttk.Entry(self.prerecorded_video_frame, width=30)
        self.video_name_entry.grid(row=0, column=1, padx=5, pady=5)

        self.browse_button = ttk.Button(self.prerecorded_video_frame, text="Browse", command=self.browse_video)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        self.show_wireframe_output_button = ttk.Button(self.prerecorded_video_frame, text="Show Wireframe Output", command=self.on_click2)
        self.show_wireframe_output_button.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

    def on_click1(self):
        detector = PoseDetector()
        cap = cv2.VideoCapture(0) 
        while True:
            success, img = cap.read()
            img = detector.findPose(img)
            cv2.imshow("Live Camera Test", img)
            c = cv2.waitKey(1)
            if c == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def on_click2(self):
        video_name1 = self.video_name_entry.get()
        if not video_name1:
            messagebox.showinfo("Error", "Please select a video :D")
            return
        detector = PoseDetector()
        cap = cv2.VideoCapture(video_name1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set the desired output frame rate
        desired_fps = 60

        # Calculate the frame rate conversion factor
        fps_conversion_factor = desired_fps / cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, desired_fps, (int(cap.get(3)), int(cap.get(4))))

        current_frame = 0
        prev_frame = None
        is_playing = True  # Variable to track play/pause state

        cv2.namedWindow("Prerecorded Video Test")

        def on_trackbar(pos):
            nonlocal current_frame
            current_frame = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        cv2.createTrackbar("Position", "Prerecorded Video Test", 0, total_frames - 1, on_trackbar)

        while True:
            if is_playing:
                success, img = cap.read()
                if success:
                    img = detector.findPose(img)

                    # Enhance colors
                    img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)  # Increase brightness and contrast
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hsv_img[:, :, 1] = hsv_img[:, :, 1] * 2.0  # Increase saturation by a factor of 2
                    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

                    if prev_frame is not None:
                        # Interpolation to increase frame rate
                        interpolated_frames = int(fps_conversion_factor)
                        for _ in range(interpolated_frames):
                            interpolated_frame = cv2.addWeighted(prev_frame, 0.3, img, 0.7, 0)  # Adjust the weights for a more popping effect
                            out.write(interpolated_frame)
                    out.write(img)
                    prev_frame = img

                    cv2.imshow("Prerecorded Video Test", img)
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cv2.setTrackbarPos("Position", "Prerecorded Video Test", current_frame)

            key = cv2.waitKey(1)
            if key == ord('p'):
                is_playing = not is_playing  # Toggle play/pause
            elif key == ord('r'):
                current_frame -= 50
                if current_frame < 0:
                    current_frame = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                cv2.setTrackbarPos("Position", "Prerecorded Video Test", current_frame)
            elif key == ord('f'):
                current_frame += 50
                if current_frame >= total_frames:
                    current_frame = total_frames - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                cv2.setTrackbarPos("Position", "Prerecorded Video Test", current_frame)
            elif key == ord('c'):
                break

            if cv2.waitKey(1) == 27:
                break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()


    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def browse_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        self.video_name_entry.delete(0, tk.END)
        self.video_name_entry.insert(0, video_path)

if __name__ == "__main__":
    main_window = MainWindow()
    main_window.mainloop()
