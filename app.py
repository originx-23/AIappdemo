import gradio as gr
import torch
import requests
from torchvision import transforms

# 如果不希望使用缓存，可以取消注释下面这一行
# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# 下载模型，并使用最新的方法加载预训练权重
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', weights='ResNet18_Weights.DEFAULT').eval()

# 从本地文件加载标签，假设你已经手动下载并保存为 "imagenet_classes.txt"
with open("imagenet_classes.txt", "r") as f:
    labels = f.read().split("\n")

def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}    
    return confidences

demo = gr.Interface(fn=predict, 
                    inputs=gr.inputs.Image(type="pil"),
                    outputs=gr.outputs.Label(num_top_classes=3),
                    examples=[["cheetah.jpg"]],
                   )
             
demo.launch()
