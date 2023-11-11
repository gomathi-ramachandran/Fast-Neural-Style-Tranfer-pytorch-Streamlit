import torch
from torchvision import transforms, models
from PIL import Image
import os

def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('RGB')

    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension

    return img_tensor

def save_as_pth(state_dict, output_folder, output_filename):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the model's state dictionary as a .pth file
    output_path = os.path.join(output_folder, output_filename)
    torch.save(state_dict, output_path)

if __name__ == "__main__":
    # Specify the path to your input image
    input_image_path = "hockney.jpg"

    # Specify the folder where you want to save the .pth file
    output_folder = "saved_models"

    # Specify the filename for the .pth file
    output_filename = "hockney.pth"

    # Load the pre-trained VGG19 model
    vgg19 = models.vgg19(pretrained=True)
    vgg19.eval()  # Set the model to evaluation mode

    # Preprocess the image
    image_tensor = preprocess_image(input_image_path)

    # Save the tensor as a .pth file in the specified folder
    save_as_pth(vgg19.state_dict(), output_folder, output_filename)

