import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SafeSteps | Fall Detection",
    page_icon="🛡️",
    layout="wide",
)

WINDOW_SIZE = 100
N_FEATURES = 9
MODEL_PATH = "cnn_transformer_model_fixed.pth"


class CNNTransformer(nn.Module):
    def __init__(self):
        super(CNNTransformer, self).__init__()
        self.cnn = nn.Conv1d(
            in_channels=9,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = CNNTransformer().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_data(uploaded_file):
    raw = pd.read_csv(uploaded_file, header=None)

    # Convert every cell to numeric; non-numeric/header rows become NaN.
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all")

    # Keep the first 9 numeric sensor columns.
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < N_FEATURES:
        raise ValueError(
            f"Expected at least {N_FEATURES} numeric sensor columns, "
            f"but found {numeric.shape[1]}."
        )

    data = numeric.iloc[:, :N_FEATURES].dropna().to_numpy(dtype=np.float32)

    if len(data) < WINDOW_SIZE:
        raise ValueError(
            f"The file contains only {len(data)} usable rows. "
            f"At least {WINDOW_SIZE} rows are required."
        )

    # Match the project notebook: fit a fresh StandardScaler per raw file.
    scaler = StandardScaler()
    data = scaler.fit_transform(data).astype(np.float32)

    # Match the notebook's non-overlapping windowing behavior.
    starts = range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE)
    windows = [data[i:i + WINDOW_SIZE] for i in starts]

    if not windows:
        raise ValueError("No complete sensor windows could be created.")

    return np.stack(windows).astype(np.float32), data


def predict(model, windows):
    device = torch.device("cpu")
    x = torch.from_numpy(windows).to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    return predictions.numpy(), probabilities.numpy()


st.title("🛡️ SafeSteps")
st.subheader("Hybrid CNN–Transformer Fall Detection")
st.write(
    "An interactive demonstration of the SafeSteps academic project using "
    "SisFall-style 9-channel wearable sensor data."
)

with st.sidebar:
    st.header("Project")
    st.write("**Model:** CNN + Transformer Encoder")
    st.write("**Input:** 9 sensor features")
    st.write("**Window:** 100 samples")
    st.write("**Classes:** Normal / Fall")
    st.divider()
    st.caption(
        "Academic demonstration only. This app is not a medical or "
        "emergency-response system."
    )

uploaded = st.file_uploader(
    "Upload a sensor CSV file",
    type=["csv"],
    help=(
        "Upload a SisFall-style file containing at least 9 numeric sensor "
        "columns. The app uses the first 9 numeric columns."
    ),
)

if uploaded is None:
    st.info(
        "Upload a CSV containing 9 sensor channels to run the trained model."
    )
    st.markdown(
        """
        **Expected format**

        - One row per sensor sample
        - 9 numeric sensor channels
        - At least 100 rows
        - SisFall-style raw sensor data works best

        The preprocessing follows the project notebook: per-file
        StandardScaler normalization followed by non-overlapping 100-sample
        windows.
        """
    )
else:
    try:
        windows, scaled_data = prepare_data(uploaded)
        model = load_model()
        predictions, probabilities = predict(model, windows)

        fall_count = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())
        total = len(predictions)

        # Overall result: majority vote across windows.
        overall_fall = fall_count > normal_count
        overall_label = "FALL DETECTED" if overall_fall else "NO FALL DETECTED"
        overall_conf = (
            probabilities[:, 1].mean()
            if overall_fall
            else probabilities[:, 0].mean()
        )

        if overall_fall:
            st.error(f"🚨 {overall_label}")
        else:
            st.success(f"✅ {overall_label}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Windows analyzed", total)
        c2.metric("Fall windows", fall_count)
        c3.metric("Normal windows", normal_count)
        c4.metric("Average confidence", f"{overall_conf * 100:.1f}%")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Sensor Signal")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(scaled_data[:, 0], label="Sensor 1")
            ax.plot(scaled_data[:, 1], label="Sensor 2")
            ax.plot(scaled_data[:, 2], label="Sensor 3")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Standardized value")
            ax.set_title("First three sensor channels")
            ax.legend()
            ax.grid(alpha=0.25)
            st.pyplot(fig, clear_figure=True)

        with right:
            st.subheader("Window Predictions")
            result_df = pd.DataFrame({
                "Window": np.arange(1, total + 1),
                "Prediction": np.where(
                    predictions == 1, "Fall", "Normal"
                ),
                "Normal %": probabilities[:, 0] * 100,
                "Fall %": probabilities[:, 1] * 100,
            })
            st.dataframe(
                result_df.style.format({
                    "Normal %": "{:.1f}",
                    "Fall %": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "The overall result is a majority vote across analyzed windows. "
            "For research use, evaluate the model on a held-out test set "
            "before making performance claims."
        )

    except Exception as e:
        st.error(f"Could not process this file: {e}")
