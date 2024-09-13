import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import json

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a trained model checkpoint.")
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(weights='DEFAULT')
    input_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(input_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 102),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

def predict(image_path, model, top_k, device):
    model.to(device)
    model.eval()
    image = process_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        probs, indices = torch.topk(torch.exp(output), top_k)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices[0].tolist()]
    return probs[0].tolist(), classes

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(args.image_path, model, args.top_k, device)
    class_names = [cat_to_name.get(cls, cls) for cls in classes]
    
    for i in range(args.top_k):
        print(f"{class_names[i]}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()
