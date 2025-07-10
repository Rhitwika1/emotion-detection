# Emotion Recognition Dataset

This repository contains a compressed dataset (`.zip`) of facial emotion images used for training and testing an emotion recognition model.

## Contents

- `images.zip` — A zipped folder containing grayscale 48x48 pixel facial images categorized by emotion classes.

Emotion categories:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Folder Structure (when unzipped)

images/
├──test
    ├── angry/
    ├── disgusted/
    ├── fearful/
    ├── happy/
    ├── neutral/
    ├── sad/
    ├── surprised/  
├──train
    ├── angry/
    ├── disgusted/
    ├── fearful/
    ├── happy/
    ├── neutral/
    ├── sad/
    ├── surprised/   

Each folder contains `.png` images representing facial expressions of that emotion.

## Use Case

These images are intended to be used with a Convolutional Neural Network (CNN) for emotion detection based on facial expressions.

This dataset is compatible with projects involving:
- Image classification
- Real-time emotion detection (webcam-based or static image)
- AI-based sentiment analysis tools
The link if the dataset is pasted.

## How to Use

After cloning or downloading the repository:

1. Unzip `images.zip`.
2. Load the images in your training script or notebook.
3. Normalize images and reshape them to `(48, 48, 1)` for CNN input.
4. Train your emotion recognition model.

