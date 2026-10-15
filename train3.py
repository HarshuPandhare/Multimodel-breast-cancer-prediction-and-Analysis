import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\Users\h1shu\OneDrive\Desktop\ML_project\data\breast-cancer-us-classification-ml-4-im-2025-2026\complete_set\training_set"

MODEL_PATH = "models/breast_cancer_efficientnet.pth"

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 15

LEARNING_RATE = 0.0001

RANDOM_SEED = 42


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("=" * 60)

print("DEVICE")

print("=" * 60)

print(device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

full_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)


print()
print("Classes:")

print(
    full_dataset.classes
)

print()

print(
    "Total images:",
    len(full_dataset)
)


# ============================================================
# SPLIT DATASET
# ============================================================

total_size = len(full_dataset)

train_size = int(
    0.70 * total_size
)

validation_size = int(
    0.15 * total_size
)

test_size = (
    total_size
    - train_size
    - validation_size
)


train_dataset, validation_dataset, test_dataset = random_split(

    full_dataset,

    [
        train_size,
        validation_size,
        test_size
    ],

    generator=torch.Generator().manual_seed(
        RANDOM_SEED
    )
)


# ============================================================
# TEST DATA TRANSFORM
# ============================================================

validation_dataset.dataset.transform = test_transform

test_dataset.dataset.transform = test_transform


print()
print("=" * 60)

print("DATA SPLIT")

print("=" * 60)

print(
    "Training:",
    len(train_dataset)
)

print(
    "Validation:",
    len(validation_dataset)
)

print(
    "Testing:",
    len(test_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=True
)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

print()
print("Loading EfficientNet-B0...")


weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# ============================================================
# FREEZE MOST BACKBONE LAYERS
# ============================================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ============================================================
# CHANGE CLASSIFIER
# ============================================================

number_of_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    number_of_features,
    2
)


model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(

    model.classifier.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_validation_accuracy = 0.0

train_losses = []

validation_losses = []

train_accuracies = []

validation_accuracies = []


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 60)

print("TRAINING")

print("=" * 60)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )


    train_accuracy = (
        correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss_total = 0.0

    validation_correct = 0

    validation_total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            validation_loss_total += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            validation_total += (
                labels.size(0)
            )


            validation_correct += (
                predicted == labels
            ).sum().item()


    validation_loss = (
        validation_loss_total
        / validation_total
    )


    validation_accuracy = (
        validation_correct
        / validation_total
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    train_losses.append(
        train_loss
    )

    validation_losses.append(
        validation_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    validation_accuracies.append(
        validation_accuracy
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )


        torch.save({

            "model_state_dict":
                model.state_dict(),

            "classes":
                full_dataset.classes,

            "image_size":
                IMAGE_SIZE

        }, MODEL_PATH)


        print(
            "Best model saved!"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 60)

print("LOADING BEST MODEL")

print("=" * 60)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)

print("TESTING")

print("=" * 60)


all_predictions = []

all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)


        _, predictions = torch.max(
            outputs,
            1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(

    all_labels,

    all_predictions

)


print()

print(
    "TEST ACCURACY:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()

print("=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)


print(

    classification_report(

        all_labels,

        all_predictions,

        target_names=full_dataset.classes,

        zero_division=0

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions

)


plt.figure(
    figsize=(6, 5)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=full_dataset.classes,

    yticklabels=full_dataset.classes

)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Breast Cancer Ultrasound Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    train_accuracies,

    label="Training Accuracy"

)


plt.plot(

    validation_accuracies,

    label="Validation Accuracy"

)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "accuracy_graph.png"
)

plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    train_losses,

    label="Training Loss"

)


plt.plot(

    validation_losses,

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "loss_graph.png"
)

plt.show()


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)

print("TRAINING AND TESTING COMPLETE")

print("=" * 60)

print()

print(
    "Best model:",
    MODEL_PATH
)

print(
    "Test accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print()

print(
    "Classes:",
    full_dataset.classes
)