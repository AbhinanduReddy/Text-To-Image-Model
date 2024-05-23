# Text-To-Image-Model
# CLIP Model for Image-Text Matching

This project implements a CLIP (Contrastive Language-Image Pretraining) model to match images with textual queries. We use a pre-trained ResNet50 as the image encoder and DistilBert as the text encoder. The model learns to project both image and text features into a shared embedding space to compute similarity.

## Project Structure

- `CFG`: Configuration settings for the model, paths, hyperparameters, etc.
- `data_preparation.py`: Script to load, preprocess, and split the data.
- `models.py`: Contains definitions for the Image Encoder, Text Encoder, and Projection Head.
- `train.py`: Contains the training and validation loop, along with utility functions.
- `inference.py`: Script to run inference and find matching images for a given query.

## Key Components

### Data Preparation
- Load image and caption data.
- Preprocess and transform images.
- Split data into training and validation sets.

### Model Architecture
- **Image Encoder**: Pre-trained ResNet50 to extract image features.
- **Text Encoder**: Pre-trained DistilBert to extract text features.
- **Projection Head**: Maps image and text features into a common embedding space.

### Training and Validation
- Train the model using contrastive loss to align image and text embeddings.
- Validate the model and save the best-performing model.

### Inference
- Load the trained model and compute image embeddings.
- Find and display matching images for a given textual query.

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Place your images in the specified path and ensure captions are available in `captions.txt`.

3. **Run Training**:
   ```bash
   python train.py
   ```

4. **Run Inference**:
   ```bash
   python inference.py
   ```

## Example Usage

After training the model, you can use it to find images that match a textual query.

```python
from inference import find_matches

# Load the model and embeddings
model, image_embeddings = get_image_embeddings(valid_df, "best.pt")

# Find and display matching images
find_matches(model, image_embeddings, query="A dog playing with a ball", image_filenames=valid_df['image'].values)
```

## Results

The model displays images that are most similar to the provided textual query, demonstrating the learned alignment between image and text embeddings.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
