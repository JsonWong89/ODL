# ODL - Lung Disease Classification

This project uses Deep Learning to classify lung diseases (Normal, Pneumonia, and Tuberculosis) from chest X-ray images.

## Project Structure
* `code/`: Source code for training and inference.
* `requirements.txt`: List of Python dependencies.
* `.gitignore`: Configured to exclude heavy `.venv`, datasets, and model weights.

## Dataset

This project requires a lung disease classification dataset with the following structure:

```
code/
 train/
    normal/
    pneumonia/
    tuberculosis/
 test/
    normal/
    pneumonia/
    tuberculosis/
 val/
     normal/
     pneumonia/
     tuberculosis/
```

** Dataset is NOT included in this repository due to size constraints.**

### Obtaining the Dataset
- Download the lung disease classification dataset from your preferred source (e.g., Kaggle, medical imaging repositories)
- Extract the images into the `train/`, `test/`, and `val/` directories as shown above
- The dataset should contain chest X-ray images categorized by disease type (normal, pneumonia, tuberculosis)

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/JsonWong89/ODL.git
cd ODL/code
```

### 2. Set up the dataset
Download and organize your dataset according to the structure shown above. Place the image directories directly in the `code/` folder.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
Refer to the Jupyter notebooks (`check_GPU.ipynb`, `load_image.ipynb`) for examples of loading and processing the data.

## Dependencies

All required Python packages are listed in `requirements.txt`. Key dependencies include:
- TensorFlow 2.20.0
- Django 5.0.9
- FastAPI
- OpenCV
- Matplotlib
- And more...

## Notes

- The image dataset directories (`train/`, `test/`, `val/`) are excluded from version control
- Sample images (`normal_lungs.jpeg`, `Pneumonia.jpeg`) are also excluded from the repository
- Make sure to set up your dataset locally before running the training scripts
