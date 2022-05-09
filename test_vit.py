from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

print(image)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", use_auth_token = True)
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
inputs = feature_extractor(images = image, return_tensors = "pt")

print(inputs["pixel_values"].shape)

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

print(last_hidden_state.shape)
print(last_hidden_state)