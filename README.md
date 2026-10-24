# Brain MRI Streamlit App

Upload an MRI → choose **ResNet50 / Custom CNN / HOG+SVM** → prediction, confidence bars, and **Grad-CAM** (deep models only).

## Files you need

```
streamlit_app/
  app.py
  requirements.txt
  models/                    ← put checkpoints here (or point sidebar path)
    resnet50_brain.pth
    cnn_brain.pth
    hog_svm_brain.joblib
```

If your checkpoints are on Google Drive (e.g. `/content/drive/MyDrive/Btumor/models`), paste that path in the app sidebar **Models folder**.

## Run locally

```bash
cd streamlit_app
pip install -r requirements.txt
# copy your .pth / .joblib into ./models/
streamlit run app.py
```

## Run in Google Colab

```python
!pip install -q streamlit torch torchvision scikit-learn scikit-image joblib opencv-python-headless pyngrok

# mount drive, then:
import os
os.environ["BTUMOR_MODEL_DIR"] = "/content/drive/MyDrive/Btumor/models"

!streamlit run app.py --server.port 8501 &

from pyngrok import ngrok
url = ngrok.connect(8501)
print(url)
```

Or use **localtunnel** / Colab’s built-in port forwarding if available.

## Notes

| Model       | Grad-CAM | Why                                      |
|-------------|----------|------------------------------------------|
| ResNet50    | Yes      | Last block of `layer4`                   |
| Custom CNN  | Yes      | Last conv layer in `features`            |
| HOG + SVM   | No       | Classical features — no CNN activations  |

Class order must match training: `glioma`, `meningioma`, `notumor`, `pituitary`.
