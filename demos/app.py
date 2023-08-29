
import gradio as gr
import os
import torch

from model import create_model
from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = ["full","half","overflowing"]

effnetb0,effnetb0_transforms = create_model(num_classes=3)

effnetb0.load_state_dict(
    torch.load(
        f="EfficientNetB0_85_epochs_0.0001_lr.pth",
        map_location=torch.device("cpu"), 
    )
)

# Create predict function
def predict(img):
  start_time=timer()
  img = effnetb0_transforms(img).unsqueeze(0)

  effnetb0.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(effnetb0(img),dim=1)
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
  pred_time = round(timer() - start_time, 5)
  return pred_labels_and_probs, pred_time

# Creating Gradio app
title = "Water Level Classification for Bottles"
description = "A computer vision model utilizing the EfficientNetB0 feature extractor to accurately classify water bottle images based on their water levels: Full, Half, or Overflowing."
article = "[More Information](https://github.com/emirhansilsupur/water-bottle-image-classification)"

example_list = [["examples/" + example] for example in os.listdir("examples")]

# Creating demo
demo = gr.Interface(fn=predict, 
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), 
                             gr.Number(label="Prediction time (s)")], 
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)
demo.launch()
