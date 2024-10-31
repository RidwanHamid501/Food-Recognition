''' Food Recognition and Nutrition Estimation '''
from os import environ, path
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
import requests
import torch
from torch import nn
from torchvision import transforms

# --- Configuration and Constants ---
DEFAULT_CLASSES = [
    'Background', 'Beef', 'Chicken', 'Broccoli', 'Rice',
    'Green beans', 'Salmon', 'Potatoes', 'Eggs', 'Carrots', 'Cucumbers'
]
DEFAULT_COLOR_MAP = np.array([
    [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
    [0, 0, 255], [255, 0, 255], [255, 255, 0], [0, 255, 255],
    [125, 0, 0], [0, 0, 125], [0, 125, 0]
])

# Set up logging for better error handling and debugging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load and Configure Classes and Color Map
CLASS_NAMES, COLOR_MAP = environ.get(
    "CLASS_NAMES", DEFAULT_CLASSES), environ.get("COLOR_MAP", DEFAULT_COLOR_MAP)


def load_config():
    """Load environment variables and configurations."""
    load_dotenv()
    api_key = environ.get("API_KEY")
    api_host = environ.get("API_HOST")
    model_path = path.join(path.dirname(__file__), '../models/model.pth')
    if not api_key or not api_host:
        logging.error("API Key or Host missing. Check environment variables.")
        raise ValueError("API Key or Host missing.")
    return api_key, api_host, model_path

# --- Utility Functions ---


def get_transform():
    """Define image transformation steps."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


def get_nutritional_info(food_item, api_key, api_host):
    """Retrieve nutritional information for a given food item."""
    url = "https://edamam-edamam-nutrition-analysis.p.rapidapi.com/api/nutrition-data"
    querystring = {"ingr": f"100g cooked {food_item}"}
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": api_host
    }
    try:
        response = requests.get(url, headers=headers,
                                params=querystring, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        logging.error(
            "Error retrieving nutritional info for %s: %s", food_item, error)

        return None


def decode_segmentation_masks(mask):
    """Decode the segmentation mask into a color image."""
    mask = mask.squeeze(0).cpu().numpy()
    color_mask = COLOR_MAP[mask]
    return color_mask.astype(np.uint8)


def display_results(original_image, color_mask):
    """Display the original image and the segmentation result side by side."""
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(color_mask)
    ax[1].set_title('Segmentation Result')
    ax[1].axis('off')
    plt.show()


def display_nutritional_details(details):
    """Helper function to display detailed nutrient information."""
    print(f"Calories: {details['calories']} kcal")
    print(f"Total Weight: {details['totalWeight']} g")
    for _, nutrient in details['totalNutrients'].items():
        print(f"{nutrient['label']}: {
              nutrient['quantity']:.2f} {nutrient['unit']}")


def display_nutritional_info(unique_classes, api_key, api_host):
    """Display nutritional info for detected food items, using helper function."""
    for cls in unique_classes[1:]:  # Skip background class
        food_item = CLASS_NAMES[cls]
        nutritional_info = get_nutritional_info(food_item, api_key, api_host)
        print(f"\nNutritional information for {food_item}:\n")
        if nutritional_info:
            display_nutritional_details(nutritional_info)
        else:
            print("Nutritional information not available.")

# --- Model Architecture ---


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Convolution, Batch Normalization, and ReLU layer."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation.

    This model uses an encoder-decoder structure with convolutional, batch normalization,
    and ReLU layers to perform pixel-wise classification on images.

    Attributes:
        encoder1, encoder2, encoder3, encoder4: Layers in the encoder path.
        decoder1, decoder2, decoder3, decoder4: Layers in the decoder path.
        pool: Max pooling layer for downsampling.
        up: Upsampling layer for increasing feature map resolution.
        out_conv: Final convolutional layer to output segmentation masks.
    """

    def __init__(self, num_classes=11):
        super().__init__()
        self.encoder1 = conv_bn_relu(3, 64)
        self.encoder2 = conv_bn_relu(64, 128)
        self.encoder3 = conv_bn_relu(128, 256)
        self.encoder4 = conv_bn_relu(256, 512)
        self.decoder1 = conv_bn_relu(512, 256)
        self.decoder2 = conv_bn_relu(256, 128)
        self.decoder3 = conv_bn_relu(128, 64)
        self.decoder4 = conv_bn_relu(64, 32)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        """Defines the forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor representing an image or batch of images.

        Returns:
            torch.Tensor: Output tensor with class predictions for each pixel in the input image.
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        dec1 = self.decoder1(self.up(enc4))
        dec2 = self.decoder2(self.up(dec1))
        dec3 = self.decoder3(self.up(dec2))
        dec4 = self.decoder4(self.up(dec3))
        return self.out_conv(dec4)

# --- Loading and Prediction Functions ---


def load_model(filepath, num_classes):
    """Load the model and use torch.jit to optimize it."""
    model = UNet(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(
            filepath, map_location='cpu', weights_only=True))
        model = torch.jit.script(model)  # Torchscript for optimized model
    except FileNotFoundError:
        logging.error("Model file not found at %s.", filepath)
        raise
    return model.eval()


def predict(image_path, model, transform, api_key, api_host):
    """Run the model on an image and print the class predictions and nutritional info."""
    original_image, image = load_and_transform_image(image_path, transform)
    prediction, color_mask = get_prediction_and_mask(
        model, image, original_image.size)
    display_results(original_image, color_mask)
    unique_classes, percentages = calculate_class_percentages(prediction)
    display_class_percentages(unique_classes, percentages)
    display_nutritional_info(unique_classes, api_key, api_host)


def load_and_transform_image(image_path, transform):
    """Load an image from path and apply transformations."""
    try:
        original_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logging.error("Image file not found at %s", image_path)
        raise
    return original_image, transform(original_image).unsqueeze(0)


def get_prediction_and_mask(model, image, original_size):
    """Generate model predictions and decode the segmentation mask."""
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)
        color_mask = decode_segmentation_masks(prediction)
    return prediction, Image.fromarray(color_mask).resize(original_size, Image.BILINEAR)


def calculate_class_percentages(prediction):
    """Calculate the percentage of each class in the prediction."""
    unique_classes, counts = torch.unique(prediction, return_counts=True)
    total_pixels = prediction.numel()
    percentages = (counts.float() / total_pixels) * 100
    return unique_classes, percentages


def display_class_percentages(unique_classes, percentages):
    """Display each class and its percentage in the prediction."""
    for cls, percentage in zip(unique_classes, percentages):
        print(f"Class {cls.item()} ({CLASS_NAMES[cls]}): {
              percentage.item():.2f}%")


# --- Main Execution ---
if __name__ == "__main__":
    API_KEY, API_HOST, MODEL_PATH = load_config()
    MODEL = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))
    TRANSFORM = get_transform()
    predict('../Test-Images/test1.jpg', MODEL, TRANSFORM, API_KEY, API_HOST)
