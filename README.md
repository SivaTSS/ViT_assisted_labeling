# ViT assisted labeling
This project uses Models like SAM and Grounding DINO to automate the labeling process
![gui_screenshot](https://github.com/SivaTSS/ViT_assisted_labeling/assets/101558717/229a7f15-34aa-4d83-b330-a66f80b202cf)

# Installation:
1. Install requirements:
```
pip install -r requirements.txt
```

2. Navigate to models folder and clone Grounding DINO repo. Alternatively, the model config file can be downloaded separately and placed at the required path.
```
cd models
git clone https://github.com/IDEA-Research/GroundingDINO.git
```
3. Navigate to weights folder and download model weights
```
cd weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

# Usage:
1. Run app.py to launch the GUI.
```
python app.py
```
2. Click on Load Image button to load an image of your choice.
3. Click on Annotate button to run Grounding DINO to get the bounding boxes and Sam to get the mask on the image. Annotations and overlays are automatically saved in the Annotation folder in VOC format.
4. Click on Quit to exit the app.

### Note:
Load Image button can be pressed anytime to load a new image. It is not required to exit the app after running on every image. 
