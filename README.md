# Lens Flare Detection

## Requirements
- Python 3
- Python packages listed in `requirements.txt`
- `jupyter` if you wish to run the notebook locally

## Installation
The required python packages are listed in `requirements.txt`. We can install them using 
```bash
$ pip install -r requirements.txt
```

## Usage

This is a two-step process.
1. Train the model using `train.py`. 
```
$ python train.py DATA_DIR
```
`DATA_DIR` is a directory containing `training`. Running this will generate our model which will be stored in `flare_detection.joblib` for use.

2. Detect lens flare via `detector.py`.
```
$ python detector.py IMG1.jpg IMG2.jpg ...
```
