from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from ImageClassifier import ImageClassifier
import os
from torch import nn;

device = "cuda" if torch.cuda.is_available() else "cpu";


# Choose a image.
custom_image_path = "/ccsopen/home/bishwodahal/python/apples_oranges/test.jpg"

import torchvision
# Load in custom image and convert the tensor values to float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1]
custom_image = custom_image / 255. 
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

os.system("timg "+custom_image_path);

# Print out image data



custom_image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)


# Loading the model
model = ImageClassifier().to(device=device);
# model = nn.DataParallel(model);
model.to(device);
state_dict = torch.load("/ccsopen/home/bishwodahal/python/apples_oranges/snapshot.pt")
model.load_state_dict(state_dict["MODEL_STATE"]);

# Adding dimension of 0 to transformed image
custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)

model.eval()
with torch.inference_mode():
    # Make a prediction on image with an extra dimension
    custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(device))

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities: {custom_image_pred_probs}")
print(f"Probability of Image being Apples: {int(custom_image_pred_probs[0][0]*100)}%")
print(f"Probability of Image being Oranges: {int(custom_image_pred_probs[0][1]*100)}%")
# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
class_names = ['Apples','Oranges']

print("The image should be of "+class_names[custom_image_pred_label.cpu()])
