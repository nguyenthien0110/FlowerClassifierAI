import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import argparse

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model checkpoint.")
    parser.add_argument('--data_dir', type=str, default='flower_data', help='Directory of the dataset')
    parser.add_argument('--arch', type=str, choices=['vgg16', 'resnet50'], default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Path to save the trained model checkpoint')
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, device, epochs):
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Training Loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation Loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

def save_checkpoint(model, optimizer, class_to_idx, save_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
    }
    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved to {save_path}')

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()]),
        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()]),
    }
    image_datasets = {x: datasets.ImageFolder(root=f"{args.data_dir}/{x}", transform=data_transforms[x]) 
                      for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) 
                   for x in ['train', 'valid']}
    
    if args.arch == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        input_features = model.classifier[0].in_features
    else:
        model = models.resnet50(weights='DEFAULT')
        input_features = model.fc.in_features
        model.fc = nn.Identity()

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(input_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_units, 102),
    )

    if args.arch == 'vgg16':
        model.classifier = classifier
    else:
        model.fc = classifier

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(), 
                           lr=args.learning_rate)

    train(model, criterion, optimizer, dataloaders, device, args.epochs)

    # Save checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_checkpoint(model, optimizer, model.class_to_idx, args.save_dir)

if __name__ == '__main__':
    main()
