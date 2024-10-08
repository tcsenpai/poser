# Poser

A Posture Detection System using computer vision and machine learning techniques.

This project implements a posture detection system using computer vision and machine learning techniques. It includes a graphical user interface for training the model, capturing pose data, and running real-time posture detection.

## Features

- Multiplatform (Windows, Linux, macOS)
- Dataset creation for good and bad postures
- Model training using custom neural network or ResNet50
- Real-time posture detection from webcam feed
- User-friendly GUI for all operations

### Future features

- Daemon mode with notifications for bad posture
- Better user experience when running the app
- Improved model accuracy

## Prerequisites

- Python 3.8+
- Conda (for environment management)
- Webcam

## Installation

1. Clone this repository and copy the env.example file to .env:
   ```
   git clone https://github.com/tcsenpai/poser.git
   cd poser
   cp env.example.env
   ```

2. Create a Conda environment:
   ```
   conda create -n posture-detection python=3.8
   conda activate posture-detection
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Activate the Conda environment:
   ```
   conda activate posture-detection
   ```

2. Run the main application:
   ```
   python main.py
   ```

3. Use the GUI to perform the following actions:
   - Set dataset and model paths
   - Capture pose data using the "Take Pose" button
   - Train the model using the captured dataset
   - Run real-time posture detection

***NOTE:***

- At the first run, the dataset directory will be created by the `take_pose.py` script (that can be either run from the main menu or from command line). You need to run the take pose option from the main menu to create the dataset and labels.
- The dataset will be saved in the `posture_dataset` directory if not specified otherwise in the .env file.
- The model will be automatically created by the `main.py` script after the dataset has been created by using the 'train' option from the main menu.

## Suggested dataset size

- Minimum: 20-100 samples of good and bad postures
- Nice: 100-500 samples of good and bad postures
- Best: 500+ samples of good and bad postures

## Project Structure

- `main.py`: The main application with GUI
- `take_pose.py`: Script for capturing pose data
- `data_loader.py`: Functions for loading and preprocessing the dataset
- `model.py`: Definition of the PostureNet model
- `train.py`: Functions for training the model
- `posture_detector.py`: Functions for real-time posture detection

## Configuration

Edit the `.env` file in the project root with the following content:

```
DATASET_PATH=path/to/your/dataset
MODEL_PATH=path/to/your/model.h5
```

Adjust the paths according to your setup.

## Credits

This project uses the following libraries:

- Tensorflow
- Keras
- OpenCV
- Pillow
- Colorama
- Numpy
- Scikit-learn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.