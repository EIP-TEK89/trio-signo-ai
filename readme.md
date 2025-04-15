# Trio-Signo AI-Service

LSF AI model recognition

# Setup

Install python **3.12.x** and run:
```sh
pip install -r requirements.txt
```

# Launch API
```sh
python main.py
```
For details about the option do:
```sh
python main.py --help
```

# Add image to the dataset

You can either manually add them in the folder `datasets/source_images/{whatever_foldze_you_want}/your_image.png`

**or**

run:
```sh
python video_recorder.py
```
Do a sign and press on the corresponding letter on the keyboard or `0`if not a sign

# Generate dataset
> Find info with `-h` parameter
```sh
python gen_traindata.py
```

# Train AI Model

```sh
python train_model.py --trainset {path_to_the_cbor_file_you_made_with_gen_traindata.py}
```

# Test AI Model

```sh
python video_recorder.py --model {path_to_the_model_you_made_with_train_model.py}
```
