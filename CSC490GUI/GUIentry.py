import cv2
import tkinter as tk
from tkinter import Text, Label, Button, Frame
from PIL import Image, ImageTk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CSC-490 Live Demo")
        
        # Main frame
        self.main_frame = Frame(root, bg="#b0bec5", padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Webcam feed
        self.video_label = Label(self.main_frame, bg="white", width=800, height=600)
        self.video_label.grid(row=0, column=0, padx=10, pady=10, rowspan=3)
        
        # Right panel
        self.right_panel = Frame(self.main_frame, bg="#b0bec5")
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        
        # First label and text area (disabled)
        self.label1 = Label(self.right_panel, text="Live Transcript", font=("Arial", 12, "bold"), bg="white")
        self.label1.pack(fill=tk.X, pady=(0, 5))
        self.text_area1 = Text(self.right_panel, width=40, height=19, state='disabled')
        self.text_area1.pack(fill=tk.X, pady=(0, 10))
        
        # Second label and text area (enabled for adding text)
        self.label2 = Label(self.right_panel, text="Model Detections", font=("Arial", 12, "bold"), bg="white")
        self.label2.pack(fill=tk.X, pady=(0, 5))
        self.text_area2 = Text(self.right_panel, width=40, height=10, state='disabled')
        self.text_area2.pack(fill=tk.X, pady=(0, 10))
        
        # Button panel
        self.button_panel = Frame(self.main_frame, bg="#b0bec5")
        self.button_panel.grid(row=2, column=1, pady=10, sticky="s")
        
        self.start_button = Button(self.button_panel, text="Start Demo", bg="#7749F8", fg="white")
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = Button(self.button_panel, text="Export Data", bg="#7749F8", fg="white")
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # Start Video Capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.update_video()
    
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        
        self.root.after(10, self.update_video)
        
    def add_text_to_text_area1(self, text):
        self.text_area1.config(state='normal')
        self.text_area1.insert(tk.END, text + "\n")
        self.text_area1.config(state='disabled')
    
    def add_text_to_text_area2(self, text):
        self.text_area2.config(state='normal')
        self.text_area2.insert(tk.END, text + "\n")
        self.text_area2.config(state='disabled')
    
    def on_closing(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    
    
    
"""
    Example text add to the text area:
    command=lambda: self.add_text_to_text_area2("Hello World!")
        *OR*
    self.add_text_to_text_area2("Hello World!")
"""