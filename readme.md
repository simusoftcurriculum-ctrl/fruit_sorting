============================================================
  FRUIT DETECTOR WITH SPEECH — Full Setup Guide
============================================================

WHAT IT DOES
------------
- Classifies fruits (fresh vs rotten) from your webcam in real time
- SPEAKS about what it sees:
    Fresh Apple  → "That is a fresh apple! Apples are crunchy and rich in fiber."
    Rotten Banana→ "Warning! This banana looks rotten. Please throw it away."
    Fresh Orange → "A fresh orange is detected! Packed with Vitamin C."

DATASET
-------
Kaggle: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
Classes: freshapples, rottenapples, freshbananas, rottenbananas, freshoranges, rottenoranges


============================================================
  STEP 0 — Install Dependencies
============================================================

Windows (recommended):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python pillow tqdm pywin32

    (If you don't have a GPU, use CPU version of torch):
    pip install torch torchvision torchaudio
    pip install opencv-python pillow tqdm pywin32

Linux:
    pip install torch torchvision torchaudio
    pip install opencv-python pillow tqdm
    sudo apt install espeak-ng

macOS:
    pip install torch torchvision torchaudio
    pip install opencv-python pillow tqdm
    (speech uses built-in 'say' command — no extra install)


============================================================
  STEP 1 — Download the Dataset
============================================================

Option A — Kaggle API (automatic):
    1. Go to https://www.kaggle.com/settings → API → Create New Token
    2. Save kaggle.json to:
       Windows: C:\Users\<YourName>\.kaggle\kaggle.json
       Linux/Mac: ~/.kaggle/kaggle.json
    3. pip install kaggle
    4. python step1_download_dataset.py

Option B — Manual download:
    1. Go to https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
    2. Click Download (free Kaggle account needed)
    3. Unzip it
    4. Make sure the folder structure looks like this:

        dataset/
          train/
            freshapples/       ← ~1500 images
            rottenapples/
            freshbananas/
            rottenbananas/
            freshoranges/
            rottenoranges/
          test/
            freshapples/       ← ~500 images
            rottenapples/
            ... (same)

    5. Place the 'dataset' folder next to the Python scripts


============================================================
  STEP 2 — Train the Model
============================================================

    python step2_train.py

This will:
  - Train MobileNetV2 on your fruits dataset for 15 epochs
  - Save the best model as:  fruit_classifier.pth
  - Save class names as:     class_names.json

Expected accuracy: ~95-98%
Training time:
  GPU → ~5-10 minutes
  CPU → ~30-60 minutes

Watch for:
  "Best model saved (val acc = 97.3%)"  ← good!


============================================================
  STEP 3 — Run Detection with Speech
============================================================

    python step3_detect_speak.py

How to use:
  - A green box appears in the centre of the screen
  - Hold a fruit inside that box
  - The model classifies it and SPEAKS about it
  - It shows FRESH ✓ or ROTTEN ✗ badge on screen

Keyboard controls:
  Q → Quit
  S → Mute speech for 10 seconds
  M → Toggle speech on/off
  F → Freeze/unfreeze the frame


============================================================
  TROUBLESHOOTING
============================================================

"No speech / silent"
  Windows → run:  pip install pywin32
             then restart and try again

"Model not found"
  → Run step2_train.py first

"Dataset folder not found"
  → Check dataset/ folder is next to the scripts
  → Check it has dataset/train/freshapples/ etc.

"Low accuracy"
  → In step2_train.py, change EPOCHS = 25

"Webcam not opening"
  → Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1)
    in step3_detect_speak.py

"torch not found"
  → pip install torch torchvision


============================================================
  FILES SUMMARY
============================================================

  step1_download_dataset.py  → downloads/verifies the dataset
  step2_train.py             → trains the fruit classifier
  step3_detect_speak.py      → webcam detection + speech
  README.txt                 → this file

  (generated after training)
  fruit_classifier.pth       → trained model weights
  class_names.json           → class names in correct order
============================================================
