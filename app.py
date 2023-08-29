# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:15:30 2023

@author: siva
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import sam
import grounding_dino
import cv2
import supervision as sv
import os

grounding_dino_model = grounding_dino.load_model()
sam_model = sam.load_model()

class ImageAnnotatorApp:
    def __init__(self, root):
        c1="#FAF1E4"
        c2="#CEDEBD"
        c3="#9EB384"
        c4="#435334"
        self.CLASSES = ['bottle', 'dog', 'person', 'nose', 'chair', 'car', 'ear']
        self.root = root
        self.root.title("Image Annotator")
        self.root.state('zoomed')
        self.root.configure(bg=c1)

        self.screen_width= self.root.winfo_screenwidth() 
        self.screen_height= self.root.winfo_screenheight() 
        self.padx = self.screen_width // 50
        self.pady = self.screen_height // 50
        self.text_size = int(self.screen_height * 0.015)
        self.button_width = int(self.screen_width * 0.08)
        self.button_height = int(self.screen_height * 0.0035)
        self.button_font = ("Helvetica", self.text_size)

        self.image_frame = tk.Frame(self.root, bd=1, bg=c1, highlightbackground=c4, relief=tk.SOLID)
        self.image_frame.grid(row=0, column=0, columnspan=3, padx=0, pady=0, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(1, weight=1)
                

        self.loaded_image_frame = tk.Frame(self.image_frame, bd=1, bg=c2, highlightbackground=c4, relief=tk.SOLID)
        self.loaded_image_frame.grid(row=0, column=0, padx=self.padx, pady=self.pady, sticky="nsew")
        self.loaded_image_label = tk.Label(self.loaded_image_frame, bg=c2)
        self.loaded_image_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.annotated_image_frame = tk.Frame(self.image_frame, bd=1, bg=c2, highlightbackground=c4, relief=tk.SOLID)
        self.annotated_image_frame.grid(row=0, column=1, padx=self.padx, pady=self.pady, sticky="nsew")
        self.annotated_image_label = tk.Label(self.annotated_image_frame, bg=c2)
        self.annotated_image_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image, width=self.button_width,
                                      height=self.button_height, bg=c3, font=self.button_font )
        self.annotate_button = tk.Button(self.root, text="Annotate", command=self.annotate_image, width=self.button_width,
                                          height=self.button_height, bg=c3, font=self.button_font )
        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit, width=self.button_width,
                                      height=self.button_height, bg=c3, font=self.button_font )

        self.loaded_image = None
        self.annotated_image = None

        self.load_button.grid(row=1, column=0, padx=self.padx, pady=self.pady)
        self.annotate_button.grid(row=1, column=1, padx=self.padx, pady=self.pady)
        self.quit_button.grid(row=1, column=2, padx=self.padx, pady=self.pady)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        
    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        self.image_name=self.file_path.rsplit("/",1)[-1][:-4]
        print("self.image_name", self.image_name)
        if self.file_path :
            image = Image.open(self.file_path)
            image.thumbnail((500, 500))  # Resize image to fit in the label
            self.loaded_image = image
            
            photo = ImageTk.PhotoImage(image)

            self.loaded_image_label.config(image=photo)
            self.loaded_image_label.image = photo  # Keep a reference to avoid garbage collection

    def annotate_image(self):
        if self.loaded_image is None:
            return
        
        SOURCE_IMAGE_PATH = self.file_path
        
        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        annotated_image = self.perform_annotation(image)
        pil_annotated_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        pil_annotated_image.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(pil_annotated_image)
         
        self.annotated_image_label.config(image=photo)
        self.annotated_image_label.image = photo

    def perform_annotation(self, image):
        detections=grounding_dino.predict(image,grounding_dino_model, self.CLASSES)
        detections=sam.predict(image,sam_model,detections)

        annotated_image1=grounding_dino.annotate_image(image, detections, self.CLASSES)
        cv2.imwrite(os.path.join("data","annotations",f"{self.image_name}_overlay_bb.png"),annotated_image1)
        annotated_image=sam.annotate_image(image, detections, self.CLASSES)
        cv2.imwrite(os.path.join("data","annotations",f"{self.image_name}_overlay_seg.png"),annotated_image)
        ANNOTATIONS_DIRECTORY = os.path.join("data", "annotations")
        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75
        sv.Dataset(classes=self.CLASSES,images={f"{self.image_name}":image},
        annotations={f"{self.image_name}":detections}).as_pascal_voc(
        annotations_directory_path=ANNOTATIONS_DIRECTORY,
        min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
        max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
        approximation_percentage=APPROXIMATION_PERCENTAGE)
        return annotated_image

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotatorApp(root)
    root.mainloop()
