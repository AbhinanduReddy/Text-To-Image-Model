import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Initialize CLIP model and tokenizer
model = CLIPModel().to(CFG.device)
model.load_state_dict(torch.load("best.pt", map_location=CFG.device))
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

# Function to extract frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

# Function to predict using CLIP model
def predict_clip(frames, query, image_filenames, n=9):
    image_embeddings = get_image_embeddings(valid_df, "best.pt")[1]
    predictions = []
    for frame in frames:
        image = Image.fromarray(frame)
        with torch.no_grad():
            image_tensor = transforms(image=image)['image'].unsqueeze(0).to(CFG.device)
            image_features = model.image_encoder(image_tensor)
            image_embeddings = model.image_projection(image_features)
            pred = find_matches(model, image_embeddings, query, image_filenames, n)
            predictions.append(pred)
    return predictions

# Main function to run the user interface
def main(video_path):
    query = input("Enter query: ")
    frames = extract_frames(video_path)
    predictions = predict_clip(frames, query, valid_df['image'].values)
    # Display the predicted results to the user
    # You can customize this part based on your preferred GUI framework
    for pred in predictions:
        plt.imshow(pred)
        plt.show()

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    main(video_path)
