# Food Recognition and Nutrition Estimation

This project provides a tool for identifying specific food items in images and estimating their nutritional information. It uses a semantic segmentation model (U-Net) to classify pixels in an image as belonging to one of several predefined food classes. Nutritional information for each detected food type is then retrieved from an external API.

## Scope

Currently, the model is configured to recognize and segment the following food items:

`['Background', 'Beef', 'Chicken', 'Broccoli', 'Rice', 'Green beans', 'Salmon', 'Potatoes', 'Eggs', 'Carrots', 'Cucumbers']`

## Features

- **Food Identification**: Detects and segments specific food items in images.
- **Nutritional Estimation**: Estimates nutritional details (e.g., calories, weight, and key nutrients) for identified food items using the Edamam Nutrition Analysis API.
- **Visualization**: Displays the original image alongside the segmentation result, with each food item represented by a distinct color.
- **Customizable**: Configuration options for API keys, model paths, and other parameters are loaded from environment variables for flexible setup.

## Setup and Installation

### Prerequisites

- Install dependencies:
  ```bash
  pip3 install -r requirements.txt
  ```

### Environment Variables
- Add the following to your .env file:

  - `API_KEY`: API key for the nutrition analysis API
  - `API_HOST`: Host URL for the API endpoint

### Model File
- Ensure the model file (`model.pth`) is stored in the specified directory (`../models/`). 
- Modify the `MODEL_PATH` in the code if needed.

## Usage

1. **Run the script:**
    ```bash
    python3 main.py
    ```

2. **Input Image**
- Change the path in the `predict` function (e.g., `../Test-Images/test1.jpg`) to test with different images.

The script will display:
- The original image and segmentation results.
- The detected food items, their proportions, and nutritional details if available.

## Key Components
- `load_model`: Loads and prepares the U-Net model for segmentation.
- `predict`: Main function to perform food recognition on an image and display results.
- `display_nutritional_info`: Retrieves and displays nutritional information for recognized food items.

## Limitations
- The model is limited to recognizing only the predefined food classes listed above.
- API responses depend on the availability and accuracy of the nutrition analysis API.