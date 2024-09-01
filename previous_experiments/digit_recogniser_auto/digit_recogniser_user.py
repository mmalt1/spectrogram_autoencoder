"""
NOTE
> ConvTranspose2d: can be seen as deconv layer
"""
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from digit_recogniser import Net

def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with the same mean and std as MNIST
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_image_class(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()

def visualize_image(image_path, predicted_class):
    image = Image.open(image_path).convert('L')
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()


device = torch.device("cpu")
model = Net().to(device)
model.load_state_dict(torch.load("5epoch_digit_rec.pt"))
img = load_and_preprocess_image("handwritten_digits/two_dark.png")
print(img.size())

predicted_digit = predict_image_class(model, img)
print("predicted digit: ", predicted_digit)

visualize_image("handwritten_digits/two_dark.png", predicted_digit)

