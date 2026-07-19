import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv

# =======================
# Config
# =======================
test_dir = "./test"
model_paths = [
    "./resnet18_best2.pth",
    "./vgg19_best6.pth",
    "./vit_base_patch16_224_best1.pth"
    "./resnet50_best33.pth",
    "./vgg16_best3.pth",
    "./vit_base_patch16_224_best8.pth"
    "./resnet34_best2.pth",
    "./vgg16_best7.pth",
    "./vit_base_patch16_224_best23.pth"
    "./resnet101_best7.pth",
    "./vgg16_best8.pth",
    "./vit_base_patch16_224_best44.pth"
    "./resnet151_best2.pth",
    "./vgg19_best5.pth",
    "./vit_base_patch16_224_best45.pth"
    "./vit_base_patch16_224_best49.pth"
]
'''
"./vit_base_patch16_224_best23.pth"
"./resnet101_best7.pth",
"./vgg16_best8.pth",
"./vit_base_patch16_224_best44.pth"
"./resnet151_best2.pth",
"./vgg19_best5.pth",
"./vit_base_patch16_224_best45.pth"
"./vit_base_patch16_224_best49.pth"
'''
# Final pneumonia 4 classes
class_names = ["Bacterial", "COVID-19", "Normal", "Virus"]
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Universal Loader (ResNet / VGG / ViT)
# =======================
def load_model(path):
    state = torch.load(path, map_location=device)
    weight_keys = list(state.keys())

    # Detect architecture
    is_resnet = any("layer4" in k for k in weight_keys)
    is_vgg = any("features" in k and "classifier" in k for k in weight_keys)
    is_vit = any("encoder" in k or "cls_token" in k for k in weight_keys)

    if is_resnet:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif is_vgg:
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif is_vit:
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        raise ValueError(f"Cannot detect model type: {path}")

    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"[Loaded] {path}")
    return model


# Load all models
models_list = [load_model(p) for p in model_paths]


# =======================
# Transform
# =======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  #transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =======================
# Voting Prediction
# =======================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    prob_sum = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for model in models_list:
            logits = model(img)
            probs = torch.softmax(logits, dim=1)
            prob_sum += probs.squeeze()

    avg_prob = prob_sum / len(models_list)
    pred_idx = torch.argmax(avg_prob).item()
    pred_label = class_names[pred_idx]

    return pred_idx, pred_label


# =======================
# Generate CSV
# =======================
output_csv = "resnet_vit_vgg_vvv.csv"

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "pred_index", "pred_label"])

    for fname in sorted(os.listdir(test_dir)):
        fpath = os.path.join(test_dir, fname)

        if os.path.isfile(fpath):
            pred_idx, pred_label = predict_image(fpath)
            writer.writerow([fname, pred_idx, pred_label])
            print(f"{fname}: {pred_idx} ({pred_label})")

print(f"\nCSV saved to {output_csv}")
