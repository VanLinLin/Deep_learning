# 1. Environment settings

## (1) Create virtual environment

```python
python -m venv .venv
```

## (2) Install requirement packages

```python
pip install -r  requirements.txt
```

# 2. Dataset prepare

## (1) Split original dataset into train, valid dataset

```bash
Data
├───train_images
│   ├───0
│   ├───1
│   ├───2
│   ├───3
│   ├───4
│   └───5
├───valid_images
│   ├───0
│   ├───1
│   ├───2
│   ├───3
│   ├───4
│   └───5
├───test.csv
```

# 3. Training

```python
python train.py [-h] [--pretrained] --data_root DATA_ROOT [--num_classes NUM_CLASSES] [--epoch EPOCH] [--visualize]
```

```bash
options:
  -h, --help            show this help message and exit
  --pretrained          Use pretrained weight.
  --data_root DATA_ROOT
                        The root path of data.
  --num_classes NUM_CLASSES
                        The number of classes of dataset. (Default: AOI dataset)
  --epoch EPOCH         Total train epochs.
  --visualize           Visualize loss, accuracy, confusion metrix, etc.
```

# 4. Testing

```python
python test.py [-h] --checkpoint_path CHECKPOINT_PATH --data_root DATA_ROOT --csv_file_path CSV_FILE_PATH [--num_classes NUM_CLASSES] [--result_save_path RESULT_SAVE_PATH]
```

```bash
options:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        THe chekcpoint path.
  --data_root DATA_ROOT
                        The root path of data.
  --csv_file_path CSV_FILE_PATH
                        The test csv file.
  --num_classes NUM_CLASSES
                        The number of classes of dataset.
  --result_save_path RESULT_SAVE_PATH
                        The number of classes of dataset.
```

# 6. Results

After training, it will create a ``runs/ResNet/{date info}``folder, contains ``logs``, ``vis`` and ``weights``.

```bash
RUNS
└───ResNet
    └───20231024183416
        ├───logs
	│   └───info.log
        ├───vis
	│   ├───confusion_matrix.jpg
	│   └───visualize.jpg
        └───weights (might have many .pth files)
	    └───minimun_valid_loss_xxx_epoch_xx.pth
```

Logs folder contains the training log, vis folder contains visualize images, including loss, accuracy, confusion matrix, etc, and weights folders contains the checkpoint file (.pth).
