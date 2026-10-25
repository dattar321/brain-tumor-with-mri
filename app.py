"""
Brain MRI Tumor Classifier — Streamlit App

Upload an MRI image, pick a model (ResNet50 / Custom CNN / HOG+SVM),
see prediction + confidence, and Grad-CAM for the deep models.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from torchvision import models, transforms
import matplotlib.pyplot as plt
import cv2

# ---------------------------------------------------------------------------
# Config — change MODEL_DIR if your checkpoints live elsewhere
# ---------------------------------------------------------------------------
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE_CNN = 224
IMG_SIZE_HOG = 128

# Default: same folder layout as the training pipeline on Drive
DEFAULT_MODEL_DIR = os.environ.get(
    "BTUMOR_MODEL_DIR",
    str(Path(__file__).resolve().parent / "models"),
)

# ---------------------------------------------------------------------------
# Model definitions (must match training)
# ---------------------------------------------------------------------------
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # inplace=False is required for Grad-CAM (inplace ReLU breaks autograd hooks)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def disable_inplace_relu(module: nn.Module) -> None:
    """Turn off inplace ReLUs so Grad-CAM backward hooks work."""
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def last_conv_layer(module: nn.Module) -> nn.Module:
    """Return the last Conv2d inside a module (or Sequential)."""
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layer found for Grad-CAM")
    return convs[-1]


# ---------------------------------------------------------------------------
# Grad-CAM (lightweight, no extra package)
# ---------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM for a single target conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, out):
        # clone() avoids view+inplace issues with hooks
        self.activations = out.detach().clone()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach().clone()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        # keep a leaf copy so backward is always allowed
        x = input_tensor.detach().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()[0]
        return cam, class_idx, probs

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def overlay_cam_on_image(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    """Resize CAM to image size and blend with a jet colormap."""
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return overlay, heatmap


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resnet(path: str, device: torch.device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    disable_inplace_relu(model)
    model.to(device).eval()
    return model


@st.cache_resource
def load_cnn(path: str, device: torch.device):
    model = BrainTumorCNN(num_classes=NUM_CLASSES)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    disable_inplace_relu(model)
    model.to(device).eval()
    return model


@st.cache_resource
def load_hog_svm(path: str):
    blob = joblib.load(path)
    return blob["pipeline"] if isinstance(blob, dict) and "pipeline" in blob else blob


def cnn_preprocess(pil_img: Image.Image) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return tf(pil_img.convert("RGB")).unsqueeze(0)


def extract_hog_features(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SIZE_HOG, IMG_SIZE_HOG))
    gray = rgb2gray(np.array(img))
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.reshape(1, -1)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain MRI Tumor Classifier")
st.caption(
    "Upload an MRI scan · choose a model · get prediction, confidence scores, and Grad-CAM (deep models)."
)

with st.sidebar:
    st.header("Settings")
    model_dir = st.text_input(
        "Models folder",
        value=DEFAULT_MODEL_DIR,
        help="Folder containing resnet50_brain.pth, cnn_brain.pth, hog_svm_brain.joblib",
    )
    model_choice = st.selectbox(
        "Model",
        ["ResNet50", "Custom CNN", "HOG + SVM"],
        help="Grad-CAM is only available for ResNet50 and Custom CNN.",
    )
    show_gradcam = st.checkbox(
        "Show Grad-CAM",
        value=True,
        disabled=(model_choice == "HOG + SVM"),
    )
    if model_choice == "HOG + SVM":
        st.info("Grad-CAM needs a CNN. HOG+SVM has no spatial activation maps.")

    st.markdown("---")
    st.markdown("**Expected files**")
    st.code(
        "resnet50_brain.pth\ncnn_brain.pth\nhog_svm_brain.joblib",
        language=None,
    )
# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_images"

sample_images = (
    sorted([
        p for p in SAMPLE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ])
    if SAMPLE_DIR.exists()
    else []
)

image_source = st.radio(
    "Choose image source",
    ["Sample images", "Upload your own"],
    horizontal=True,
)

pil_img = None

if image_source == "Sample images":
    if sample_images:
        selected_image = st.selectbox(
            "Select a sample MRI image",
            sample_images,
            format_func=lambda p: p.name,
        )
        pil_img = Image.open(selected_image).convert("RGB")
    else:
        st.warning("No sample images found in the sample_images folder.")
        st.stop()

else:
    uploaded = st.file_uploader(
        "Upload MRI image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )

    if uploaded is None:
        st.info("Upload an image to start.")
        st.stop()

    pil_img = Image.open(uploaded).convert("RGB")

col_img, col_out = st.columns([1, 1.2])

with col_img:
    st.subheader("Input")
    st.image(pil_img, use_container_width=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = Path(model_dir)

# Resolve checkpoint paths
paths = {
    "ResNet50": model_dir / "resnet50_brain.pth",
    "Custom CNN": model_dir / "cnn_brain.pth",
    "HOG + SVM": model_dir / "hog_svm_brain.joblib",
}
ckpt_path = paths[model_choice]

if not ckpt_path.exists():
    st.error(f"Checkpoint not found:\n`{ckpt_path}`\n\nUpdate **Models folder** in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
try:
    if model_choice == "HOG + SVM":
        pipe = load_hog_svm(str(ckpt_path))
        feat = extract_hog_features(pil_img)
        pred_id = int(pipe.predict(feat)[0])
        # decision_function → rough scores; softmax-like via distance ranking
        if hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(feat)
            if scores.ndim == 1:
                # binary edge case
                probs = np.array([1 - scores[0], scores[0]])
            else:
                # multiclass: convert distances to pseudo-probabilities
                e = np.exp(scores - scores.max())
                probs = (e / e.sum()).ravel()
        else:
            probs = np.zeros(NUM_CLASSES)
            probs[pred_id] = 1.0
        pred_name = CLASSES[pred_id]
        cam = None
        heatmap = None
        overlay = None

    else:
        if model_choice == "ResNet50":
            model = load_resnet(str(ckpt_path), device)
            target_layer = last_conv_layer(model.layer4)
        else:
            model = load_cnn(str(ckpt_path), device)
            # last Conv2d in features (not the following ReLU/MaxPool)
            target_layer = last_conv_layer(model.features)

        tensor = cnn_preprocess(pil_img).to(device)

        if show_gradcam:
            cam_helper = GradCAM(model, target_layer)
            try:
                cam, pred_id, probs = cam_helper.generate(tensor)
                overlay, heatmap = overlay_cam_on_image(pil_img, cam)
            finally:
                cam_helper.close()
        else:
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_id = int(probs.argmax())
            cam = heatmap = overlay = None

        pred_name = CLASSES[pred_id]

except Exception as e:
    st.exception(e)
    st.stop()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
with col_out:
    st.subheader("Prediction")
    st.markdown(f"### **{pred_name}**")
    st.metric("Confidence", f"{probs[pred_id] * 100:.1f}%")

    st.markdown("#### Class probabilities")
    order = np.argsort(probs)[::-1]
    for i in order:
        st.progress(float(probs[i]), text=f"{CLASSES[i]} — {probs[i]*100:.1f}%")

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = ["#2ecc71" if i == pred_id else "#3498db" for i in range(NUM_CLASSES)]
    ax.barh(CLASSES, probs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

if show_gradcam and overlay is not None:
    st.markdown("---")
    st.subheader("Grad-CAM")
    st.caption(
        "Highlights regions the network relied on most for this prediction "
        "(warmer = higher contribution)."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(pil_img, caption="Original", use_container_width=True)
    with c2:
        st.image(heatmap, caption="Heatmap", use_container_width=True)
    with c3:
        st.image(overlay, caption="Overlay", use_container_width=True)

st.markdown("---")
st.caption(f"Device: `{device}` · Checkpoint: `{ckpt_path}`")
