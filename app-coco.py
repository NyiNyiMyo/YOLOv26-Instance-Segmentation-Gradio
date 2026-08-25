# app.py
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import spaces

model = YOLO('yolo26s-seg.pt')
modelsem = YOLO('yolo26s-sem.pt')

@spaces.GPU
def predict_image(img):
    results = model.predict(source=img, save=False)
    semresults = modelsem.predict(source=img, save=False)
    semresults_image_bgr = semresults[0].plot()
    annotated_image_bgr = results[0].plot(img=semresults_image_bgr)
    
    annotated_image_rgb = annotated_image_bgr[..., ::-1]
    return Image.fromarray(annotated_image_rgb)

nyiapp = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Results Image"),
    title="COCO Panoptic Segmentation Fusion by YOLOv26",
    description="YOLOv26 Fusion of Instance & Semantic Segmentation Inference.",

    examples=["sample1.jpg", "sample2.jpg", "sample3.jpg", "sample4.jpg"]
)

nyiapp.launch()