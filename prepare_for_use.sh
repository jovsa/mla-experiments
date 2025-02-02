#!/bin/bash

echo "Starting setup process..."

echo "Installing Python requirements..."
python3 -m pip install -r requirements.txt

echo "Changing to data directory..."
cd ./data/

echo "Downloading and preparing data..."
python3 download_data.py
python3 hftokenizer.py
python3 construct_dataset.py

echo "Copying tokenizer files..."
cp -R hftokenizer ../

echo "Creating necessary directories..."
cd ../
mkdir -p ./figures
mkdir -p ./weights

echo "Starting model training..."
python3 train_model.py

echo "Setup complete!"
