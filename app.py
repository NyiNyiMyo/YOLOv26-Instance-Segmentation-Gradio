# app.py
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import spaces

model = YOLO('best.pt')

@spaces.GPU
def predict_image(img):
    results = model.predict(source=img, save=False)
    annotated_image_bgr = results[0].plot()
    annotated_image_rgb = annotated_image_bgr[..., ::-1]
    return Image.fromarray(annotated_image_rgb)

nyiapp = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Results Image"),
    title="Multi SIS Instance Segmentation by YOLOv26",
    description="YOLOv26 Instance Segmentation Inference for laparoscopic surgical instruments."
)

nyiapp.launch()