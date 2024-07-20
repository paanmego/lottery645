import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import ast

MODEL_PATH = 'lotto_rnn_model.pth'
SCALER_PATH = 'lotto_scaler.save'

class LottoDataset(Dataset):
    def __init__(self, data, n_steps):
        self.data = data
        self.n_steps = n_steps

    def __len__(self):
        return max(len(self.data) - self.n_steps, 0)  # 음수가 되지 않도록 수정

    def __getitem__(self, idx):
        if idx + self.n_steps >= len(self.data):
            x = self.data[idx:]
            y = self.data[-1]  # 마지막 데이터를 타겟으로 사용
        else:
            x = self.data[idx:idx+self.n_steps]
            y = self.data[idx+self.n_steps]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class LottoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LottoRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(data, n_steps=10, epochs=100, batch_size=32):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    dataset = LottoDataset(scaled_data, n_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LottoRNN(input_size=6, hidden_size=50, output_size=6)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

def fine_tune_model(new_data):
    model = LottoRNN(input_size=6, hidden_size=50, output_size=6)
    model.load_state_dict(torch.load(MODEL_PATH))
    scaler = joblib.load(SCALER_PATH)
    
    # 기존 데이터 로드
    data = pd.read_csv('lotto_data.csv')
    all_numbers = np.array([ast.literal_eval(num) for num in data['numbers']])
    
    # 새 데이터를 기존 데이터에 추가
    all_numbers = np.vstack((all_numbers, new_data))
    
    scaled_data = scaler.transform(all_numbers)
    
    n_steps = min(10, len(scaled_data) - 1)  # n_steps를 데이터 길이에 맞게 조정
    dataset = LottoDataset(scaled_data, n_steps)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), MODEL_PATH)

def predict_next_numbers():
    model = LottoRNN(input_size=6, hidden_size=50, output_size=6)
    model.load_state_dict(torch.load(MODEL_PATH))
    scaler = joblib.load(SCALER_PATH)
    
    data = pd.read_csv('lotto_data.csv')
    # 'numbers' 열의 문자열을 실제 리스트로 변환
    numbers = np.array([ast.literal_eval(num) for num in data['numbers']])
    
    scaled_data = scaler.transform(numbers)
    last_10_draws = torch.FloatTensor(scaled_data[-10:]).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        prediction = model(last_10_draws)
    
    prediction = scaler.inverse_transform(prediction.numpy())
    return np.round(prediction[0]).astype(int)

def initialize_or_update_model(data):
    if not os.path.exists(MODEL_PATH):
        train_model(data)
    else:
        fine_tune_model(data[-1:])

if __name__ == "__main__":
    data = pd.read_csv('lotto_data.csv')
    # 'numbers' 열의 문자열을 실제 리스트로 변환
    numbers = np.array([ast.literal_eval(num) for num in data['numbers']])
    
    initialize_or_update_model(numbers)
    print("Next predicted numbers:", predict_next_numbers())