import serial.tools.list_ports
from pyOpenBCI import OpenBCICyton
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

SAMPLING_RATE = 250
BANDPASS_LOW = 1
BANDPASS_HIGH = 40
raw_eeg_window = []
classes = ['left', 'right', 'up', 'down']

def bandpass_filter(data, low, high, fs):
    b, a = butter(N=4, Wn=[low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=0)

def preprocess_live_eeg(raw_window):
    """
    raw_window: np.array of shape (250, 16)
    returns: processed_window of shape (250, 15)
    """
    assert raw_window.shape == (250, 16), "Expected raw EEG shape (250, 16)"

    raw_window = np.delete(raw_window, 11, axis=1)  # → (250, 15)

    filtered = bandpass_filter(raw_window, BANDPASS_LOW, BANDPASS_HIGH, SAMPLING_RATE)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(filtered)

    return normalized

class EEG_GRU(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
        super(EEG_GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)
    
def inference(raw_eeg_window: np.ndarray, model_path="./model/eeg_gru_model_2.0.pth"):
    processed = preprocess_live_eeg(raw_eeg_window)

    checkpoint = torch.load(model_path, map_location='cpu')
    model = EEG_GRU(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        x = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)  # (1, 250, 15)
        output = model(x)
        pred = torch.argmax(output, dim=1).item()

    return pred

def main(sample):
    global raw_eeg_window

    raw_eeg_window.append(sample.channels_data)  # sample.channels_data → list of 16 values

    if len(raw_eeg_window) == 250:
        raw_array = np.array(raw_eeg_window)  # shape (250, 16)
        pred = inference(raw_array)
        print("Predicted class:", classes[pred])
        raw_eeg_window = []  # reset for next window

def print_raw(sample):
    print(sample.channels_data)

def find_openbci_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "usbserial" in port.device.lower() or "FTDI" in port.description:
            return port.device
    raise IOError("OpenBCI port not found. Make sure the board is connected.")

board = OpenBCICyton(port=find_openbci_port(), daisy=True)
board.start_stream(main)