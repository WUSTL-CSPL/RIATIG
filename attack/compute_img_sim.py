import torch
import clip
from PIL import Image

def compute_img_sim(img1_path, img2_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
    image_2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        img1_features = model.encode_image(image_1)
        img2_features = model.encode_image(image_2)

        img1_features /= img1_features.norm(dim=-1, keepdim=True)
        img2_features /= img2_features.norm(dim=-1, keepdim=True)

        similarity = 100. * (img1_features @ img2_features.T)
    
    return similarity 