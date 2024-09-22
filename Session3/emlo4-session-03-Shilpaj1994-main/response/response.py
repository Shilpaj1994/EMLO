from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from PIL import Image
import torch
from io import BytesIO
from typing import List
from model import Net

app = FastAPI()

# Load the model
model = Net()
model.load_state_dict(torch.load("/opt/mount/model/mnist_cnn.pt"))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        image_data = await file.read()
        # print(image_data)
        image = Image.open(BytesIO(image_data)).convert('L')
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")

    with torch.no_grad():
        prediction = model(image)
        predicted_class = torch.argmax(prediction, dim=1)

    return {"class": int(predicted_class)}