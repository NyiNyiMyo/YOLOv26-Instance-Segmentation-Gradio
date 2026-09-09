import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image
from ultralytics import YOLO
import spaces

model = YOLO('yolo26s-seg.pt')
modelsem = YOLO('yolo26s-sem.pt')

IMAGE_SIZE_CHOICES = [320, 512, 640]
CUSTOM_CSS = (Path(__file__).parent / "makers.css").read_text()

@spaces.GPU
def predict_image(img, conf_threshold, iou_threshold, show_inst, show_sem, imgsz):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
        save=False
    )
    semresults = modelsem.predict(source=img, save=False)
    semresults_image_bgr = semresults[0].plot()
    annotated_image_bgr = results[0].plot(img=semresults_image_bgr)
    
    annotated_image_rgb = annotated_image_bgr[..., ::-1]
    return Image.fromarray(annotated_image_rgb)

@spaces.GPU
def predict_video(video_path, conf_threshold, iou_threshold, show_inst, show_sem, imgsz):
    if video_path is None:
        return None
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_output.name
    temp_output.close()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            save=False
        )
        semresults = modelsem.predict(source=frame, save=False)
        semresults_image_bgr = semresults[0].plot()
        annotated_image_bgr = results[0].plot(img=semresults_image_bgr)
        
        out.write(annotated_image_bgr)

    cap.release()
    out.release()

    return output_path

@spaces.GPU
def predict_webcam(frame, conf_threshold, iou_threshold, show_inst, show_sem, imgsz):
    if frame is None:
        return None

    if isinstance(frame, np.ndarray):
        # Gradio webcam sends RGB, but Ultralytics YOLO expects BGR for OpenCV operations
        # Convert RGB to BGR for YOLO
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run inference
        results = model.predict(
            source=frame_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            save=False
        )
        semresults = modelsem.predict(source=frame, save=False)
        semresults_image_bgr = semresults[0].plot()
        annotated_image_bgr = results[0].plot(img=semresults_image_bgr)

        # Convert BGR to RGB for Gradio
        return cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

    return None

# Create the Gradio app with tabs
with gr.Blocks(title="COCO Panoptic Segmentation Fusion by YOLOv26 🚀") as nyiapp:
    gr.Markdown(
        """
<div align="center">
    <h1>COCO Panoptic Segmentation Fusion by YOLOv26 🚀 </h1>
</div>
YOLOv26 Fusion of Instance & Semantic Segmentation Inference.

"""
    )
    with gr.Tabs():
        # Image Tab
        with gr.TabItem("📷 Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload an Image")
                    img_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    img_iou = gr.Slider(minimum=0, maximum=1, value=0.3, label="IoU threshold")
                    img_inst = gr.Checkbox(value=True, label="Show Instance results")
                    img_sem = gr.Checkbox(value=True, label="Show Semantic results")
                    img_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                    img_btn = gr.Button("Run inference", variant="primary")
                with gr.Column():
                    img_output = gr.Image(type="pil", label="Results Image")

            img_btn.click(
                predict_image,
                inputs=[img_input, img_conf, img_iou, img_inst, img_sem, img_size],
                outputs=img_output,
            )

            gr.Examples(
                examples=[
                    ["sample1.jpg", 0.25, 0.3, True, True, 640],
                    ["sample2.jpg", 0.25, 0.3, True, True, 640],
                    ["sample3.jpg", 0.25, 0.3, True, True, 640],
                    ["sample4.jpg", 0.25, 0.3, True, True, 640],
                ],
                inputs=[img_input, img_conf, img_iou, img_inst, img_sem, img_size],
                outputs=img_output,
                fn=predict_image,
            )

        # Video Tab
        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload short Video")
                    vid_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    vid_iou = gr.Slider(minimum=0, maximum=1, value=0.3, label="IoU threshold")
                    vid_inst = gr.Checkbox(value=True, label="Show Instance results")
                    vid_sem = gr.Checkbox(value=True, label="Show Semantic results")
                    vid_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                    vid_btn = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Results Video")

            vid_btn.click(
                predict_video,
                inputs=[vid_input, vid_conf, vid_iou, vid_inst, vid_sem, vid_size],
                outputs=vid_output,
            )

        # Webcam Tab - Real-time streaming
        with gr.TabItem("📹 Webcam"):
            gr.Markdown("### Real-time Webcam Inference")
            gr.Markdown("Streaming for live inference!")
            with gr.Row():
                with gr.Column():
                    webcam_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    webcam_iou = gr.Slider(minimum=0, maximum=1, value=0.3, label="IoU threshold")
                    webcam_inst = gr.Checkbox(value=True, label="Show Instance results")
                    webcam_sem = gr.Checkbox(value=True, label="Show Semantic results")
                    webcam_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                with gr.Column():
                    # Streaming webcam input with real-time output
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam (streaming)",
                        streaming=True,
                    )
                    webcam_output = gr.Image(type="numpy", label="Results Live")

            # Stream event for real-time detection
            webcam_input.stream(
                predict_webcam,
                inputs=[webcam_input, webcam_conf, webcam_iou, webcam_inst, webcam_sem, webcam_size],
                outputs=webcam_output,
            )

nyiapp.launch(css=CUSTOM_CSS)
