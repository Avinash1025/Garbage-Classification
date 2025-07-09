import torch
from PIL import Image
from utils.transforms import get_transforms

def predict_image(model, image_path, class_names):
    transform = get_transforms(train=False)

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = class_names[torch.argmax(output).item()]
    return predicted_class
