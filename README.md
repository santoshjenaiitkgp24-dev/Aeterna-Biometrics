# Facial Age & Gender Prediction — Multi-Task CNN

> Predicts a person's age (regression) and gender (classification) simultaneously from a single face image using a shared ResNet50 backbone with task-specific output heads.

---

## Overview / Problem Statement

Estimating age and gender from facial images is a structured multi-task learning problem — both targets come from the same input (a face), but require different output types: age is a continuous variable (regression) and gender is binary (classification).

A naive approach trains two separate models. This project instead uses a **single shared backbone** (ResNet50) with two dedicated output heads, allowing both tasks to benefit from shared facial feature representations while being optimised independently. The model is evaluated using a custom harmonic mean metric that penalises models that are strong on one task but weak on the other — forcing genuinely balanced performance.

---

## Dataset

- **Source:** [Facial Recognition App Dataset](https://www.kaggle.com/datasets/santoshhhhh/facial-recognition-app-dataset) (Kaggle)
- **Size:** ~40,000 face images (JPEG, RGB)
- **Train CSV columns:** `id`, `full_path`, `gender`, `age`
- **Targets:**
  - `age` — integer, continuous (regression target)
  - `gender` — binary: 0 or 1 (classification target)
- **Train/Val split:** 80/20 using `sklearn.train_test_split` (random_state=42)
- **Test set:** Blind — no labels provided; predictions submitted as CSV

> **Note:** Gender class balance and full age distribution were not analysed during this project. This is a known gap — class imbalance in gender or skewed age distribution could affect model performance and is worth investigating before deployment.

---

## Methodology

### 1. Data Pipeline

Images are loaded lazily using `tf.data` — paths and labels are stored in memory, images are read from disk only at training time. This avoids loading 40K images into RAM at once.

```
train.csv → path list + label arrays
         → tf.data.Dataset.from_tensor_slices
         → map(preprocess) [parallel, AUTOTUNE]
         → shuffle(1000) → batch(32) → prefetch(AUTOTUNE)
```

### 2. Augmentation

Applied **only to training data** — validation images are only resized and normalised.

| Transform | Value | Reasoning |
|---|---|---|
| Random brightness | ±0.1 | Simulates different lighting conditions |
| Random contrast | 0.9–1.1 | Simulates camera variation |
| Horizontal flip | 50% | Faces are bilaterally symmetric |

Normalisation: pixel values divided by 255.0 → range [0, 1].

### 3. Model Architecture

```
Input (224×224×3)
     │
ResNet50 (ImageNet pretrained, include_top=False)
     │
GlobalAveragePooling2D  →  2048-dim vector
     │
Dense(256, relu)
     │
Dropout(0.4)
     ┌──────────────┴──────────────┐
Dense(1, linear)          Dense(1, sigmoid)
  age_output               gender_output
```

- **Backbone:** ResNet50 pretrained on ImageNet. `include_top=False` removes the original 1000-class classification head, retaining the feature extractor.
- **GlobalAveragePooling2D:** Converts the 7×7×2048 feature map to a single 2048-dim vector — avoids flattening which would produce a 100K-dim vector.
- **Dropout(0.4):** Regularisation to prevent overfitting on the shared representation.
- **age_output:** Linear activation — unconstrained output suitable for regression over a continuous range.
- **gender_output:** Sigmoid activation — outputs a probability in [0, 1], thresholded at 0.5 for binary prediction.

### 4. Training Strategy

**Two-stage fine-tuning:**

| Stage | Backbone | LR | Epochs | Purpose |
|---|---|---|---|---|
| Stage 1 | Frozen | 0.001 | 10 | Train new heads without corrupting pretrained weights |
| Stage 2 | Unfrozen | 0.00001 | 15 | Fine-tune full model at low LR for face-specific features |

Freezing the backbone in Stage 1 prevents the randomly-initialised output heads from producing large gradients that destroy the ImageNet-learned weights before the heads have stabilised.

### 5. Loss Functions & Optimisation

```python
loss = {
    'age_output'   : combined_mae_mse,      # 0.5*MAE + 0.5*MSE
    'gender_output': binary_crossentropy
}
loss_weights = {'age_output': 1.0, 'gender_output': 1.0}
optimizer = Adam
```

- **Combined MAE+MSE for age:** Pure MSE penalises large age errors quadratically — predicting 20 when true age is 60 incurs a penalty of 1600. MAE penalises it linearly at 40. The 50/50 blend is more robust to outlier ages while still maintaining sensitivity to moderate errors.
- **Binary crossentropy for gender:** Standard choice for binary classification with sigmoid output.
- **Equal loss weights:** Age is the harder task — giving it the same weight as gender (rather than the original 0.5) forces the optimiser to prioritise reducing age error.

### 6. Evaluation Metric

The competition uses a custom harmonic mean score:

```
age_score   = 1 - min(RMSE(age), 30) / 30
gender_score = Macro F1

final_score = 2 × (age_score × gender_score) / (age_score + gender_score)
```

- **age_score** is clamped at RMSE=30 (worst possible = 0, best = 1)
- **Macro F1** averages F1 across both gender classes equally — penalises bias toward the majority class
- **Harmonic mean** ensures a model cannot score well by excelling at one task and ignoring the other

A custom Keras callback computes and prints all three scores at the end of every epoch.

---

## Results

Evaluated on the held-out validation set (20% of training data, ~8,000 images):

### Age Regression

| Metric | Value |
|---|---|
| Mean Squared Error (MSE) | 77.85 |
| Mean Absolute Error (MAE) | 6.45 years |
| Root Mean Squared Error (RMSE) | 8.82 years |
| Age Score (0 to 1) | 0.7059 |

### Gender Classification

| Metric | Value |
|---|---|
| Accuracy | 89.80% |
| Macro F1-Score | 0.8498 |

### Final Competition Score

| Metric | Value |
|---|---|
| **Harmonic Mean (age_score + gender_score)** | **0.7712** |

The harmonic mean of 0.7712 reflects genuinely balanced performance across both tasks — an age score of 0.706 (RMSE ~8.8 years) and a gender Macro F1 of 0.850, combined via the competition's harmonic mean formula that penalises task imbalance.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.x | Model building, training, tf.data pipeline |
| ResNet50 | via `tf.keras.applications` | Pretrained backbone |
| NumPy | — | Array operations, metric calculation |
| Pandas | — | CSV loading and label extraction |
| scikit-learn | — | Train/val split, F1 score calculation |
| Matplotlib | — | Training visualisation |
| Kaggle | — | Dataset access, notebook environment |

---

## How to Run

### On Kaggle (recommended)

1. Fork the notebook at the link above
2. Attach the dataset `santoshhhhh/facial-recognition-app-dataset`
3. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
4. Run All

### Locally

```bash
# Install dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib

# Dataset must be downloaded from Kaggle first
kaggle datasets download santoshhhhh/facial-recognition-app-dataset
unzip facial-recognition-app-dataset.zip -d ./face_dataset

# Update base_path in the notebook to point to ./face_dataset/
# Then run cells in order
```

### Generating Submission

The final cell constructs `submission.csv` automatically after prediction:

```python
# Predictions output
age_preds    = predictions[0].flatten()          # continuous float
gender_preds = (predictions[1] > 0.5).astype(int).flatten()  # 0 or 1

submission['age']    = age_preds
submission['gender'] = gender_preds
submission.to_csv('submission.csv', index=False)
```

---

## Limitations & Future Work

### Current Limitations

- **No face detection preprocessing:** The model assumes the input image is already a cropped face. In production, an upstream face detector (e.g. MTCNN or RetinaFace) would be required.
- **No prediction confidence:** The model outputs a single point estimate for age with no uncertainty measure. A prediction of "age: 25" and "age: 72" look equally confident.
- **Unknown demographic bias:** Age distribution and gender class balance in the training data were not analysed. If the dataset skews toward certain age groups or demographics, predictions will be systematically worse for underrepresented groups.
- **Normalisation deviation:** Pixel values divided by 255.0 rather than using ResNet50's official channel-wise mean subtraction preprocessor — a minor deviation that may slightly reduce backbone performance.

### Future Work

- **EfficientNetB3 backbone:** Compound-scaled architecture likely to generalise better on a 40K dataset with fewer parameters than ResNet50
- **Ordinal regression for age (CORAL loss):** Encodes the rank-consistency of age directly into the loss — prediction 25 is constrained to be closer to 26 than to 60 at the loss level, not just implicitly
- **CBAM attention module:** Lightweight spatial + channel attention inserted after the backbone, teaching the model to focus on wrinkle/jaw/eye regions for age estimation
- **albumentations augmentation:** More aggressive, face-aware augmentation pipeline to improve generalisation
- **Age distribution analysis:** EDA on age histogram to identify underrepresented decades and apply oversampling or loss reweighting accordingly
- **TFLite conversion:** For real-time inference at edge or mobile deployment
