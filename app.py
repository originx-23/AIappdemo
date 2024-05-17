import gradio as gr
import torch
from torchvision import models, transforms

# 加载本地预训练模型
model = models.resnet18()
model.load_state_dict(torch.load('./resnet18-5c106cde.pth'))
model.eval()

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
