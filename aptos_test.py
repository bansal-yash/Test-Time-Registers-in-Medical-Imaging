import sys
import os
from tqdm import tqdm
import pandas as pd
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import argparse
from shared.utils import filter_layers

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

sys.path.append("clip/")
sys.path.append("dinov2/")

parser = argparse.ArgumentParser(description="Train/Test with specified model")
parser.add_argument(
    "--model",
    type=str,
    default="clip",
    choices=["clip", "dinov2"],
    help="Model to use (clip or dinov2)",
)
parser.add_argument(
    "--image_path",
    type=str,
    default="/Users/yash/Desktop/orthonet-data",
    help="Path to the image dataset directory",
)
args = parser.parse_args()

MODEL = args.model

if MODEL == "dinov2":
    from dinov2_state import load_dinov2_state

    config = {
        "backbone_size": "vitl14",
        "device": "mps",
        "detect_outliers_layer": -2,
        "register_norm_threshold": 150,
        "highest_layer": 19,
        "top_k": 50,
    }

    state = load_dinov2_state(config)

elif MODEL == "clip":
    from clip_state import load_clip_state

    config = {
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "device": "mps",
        "highest_layer": 5,
        "detect_outliers_layer": -1,
        "register_norm_threshold": 30,
        "top_k": 20,
    }

    state = load_clip_state(config)


IMAGE_PATH = args.image_path
IMAGE_SIZE = 224

run_model = state["run_model"]
model = state["model"]
preprocess = state["preprocess"]
hook_manager = state["hook_manager"]
num_layers = state["num_layers"]
num_heads = state["num_heads"]
patch_size = state["patch_size"]
config = state["config"]
patch_height = IMAGE_SIZE // patch_size
patch_width = IMAGE_SIZE // patch_size
device = "mps"


register_neurons = torch.load(f"Trained_Models/aptos_{MODEL}/register_neurons.pt")

top_k = config["top_k"]
highest_layer = config["highest_layer"]
num_registers = 1

filtered_register_neurons = filter_layers(register_neurons, highest_layer=highest_layer)

neurons_to_ablate = dict()
for layer, neuron, score in filtered_register_neurons[:top_k]:
    if layer not in neurons_to_ablate:
        neurons_to_ablate[layer] = []
    neurons_to_ablate[layer].append(neuron)
print(neurons_to_ablate)


test_df = pd.read_csv(f"{IMAGE_PATH}/test.csv")
test_img_dir = f"{IMAGE_PATH}/test_images/test_images"

# Get classes and create mapping
classes = sorted(test_df["diagnosis"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)


class AptosDataset(Dataset):
    def __init__(
        self, dataframe, img_dir, class_to_idx, model, ttr=False, max_images=None
    ):
        self.dataframe = dataframe.reset_index(drop=True)

        # Limit the number of images if max_images is specified
        if max_images is not None and max_images < len(self.dataframe):
            self.dataframe = self.dataframe.sample(
                n=max_images, random_state=42
            ).reset_index(drop=True)

        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.model = model
        self.device = device
        self.ttr = ttr

        # Set model to eval mode
        self.model.eval()
        self.model.to(device)

        self.representations = []
        self.labels = []

        with torch.no_grad():
            for _, row in tqdm(
                self.dataframe.iterrows(),
                total=len(self.dataframe),
                desc="Processing images",
            ):
                # Changed: use id_code with .png extension
                img_filename = row["id_code"] + ".png"
                img_path = os.path.join(self.img_dir, img_filename)
                img = Image.open(img_path).convert("RGB")
                processed_image = preprocess(img).unsqueeze(0).to(device)

                hook_manager.reinit()
                if ttr:
                    hook_manager.intervene_register_neurons(
                        neurons_to_ablate=neurons_to_ablate,
                        num_registers=num_registers,
                        normal_values="zero",
                        scale=1,
                    )
                hook_manager.finalize()

                representation = run_model(
                    model, processed_image, num_registers=num_registers
                )
                representation = representation.squeeze(0).cpu()

                self.representations.append(representation)
                # Changed: use "diagnosis" column instead of "labels"
                self.labels.append(self.class_to_idx[row["diagnosis"]])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        representation = self.representations[idx]
        label = self.labels[idx]

        return representation, label


test_dataset_no_ttr = AptosDataset(
    test_df, test_img_dir, class_to_idx, model, ttr=False, max_images=None
)
test_dataset_ttr = AptosDataset(
    test_df, test_img_dir, class_to_idx, model, ttr=True, max_images=None
)

test_loader_no_ttr = DataLoader(
    test_dataset_no_ttr, batch_size=32, shuffle=True, num_workers=0
)
test_loader_ttr = DataLoader(
    test_dataset_ttr, batch_size=32, shuffle=True, num_workers=0
)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}
    metrics["accuracy"] = (y_true == y_pred).mean()

    top3 = np.argsort(-y_prob, axis=1)[:, :3]
    top3_acc = np.mean([y_true[i] in top3[i] for i in range(len(y_true))])
    metrics["top3_accuracy"] = top3_acc

    metrics["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except:
        metrics["auc_roc"] = None

    return metrics


def test(model, test_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))

    print(f"Test Acc ={metrics['accuracy']:.4f}")

    return metrics


print(device)

if MODEL == "clip":
    input_dim = 512
else:
    input_dim = 1024

# Usage example:
classifier = LinearClassifier(input_dim=input_dim, num_classes=num_classes)
classifier = classifier.to(device)

print(classifier)


method = "no_ttr"
classifier.load_state_dict(
    torch.load(
        f"Trained_Models/aptos_{MODEL}/best_model_" + method + ".pth",
        map_location=device,
    )
)

test_metrics = test(classifier, test_loader_no_ttr)
classifier = LinearClassifier(input_dim=input_dim, num_classes=num_classes)
classifier = classifier.to(device)

print(classifier)


method = "ttr"
classifier.load_state_dict(
    torch.load(
        f"Trained_Models/aptos_{MODEL}/best_model_" + method + ".pth",
        map_location=device,
    )
)

test_metrics = test(classifier, test_loader_ttr)
