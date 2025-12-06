# Revised code: GPU-focused + XGBoost + CatBoost (no sklearn CPU models)
# ---------------------------------------------------------------
# - PyTorch NN runs fully on GPU
# - Feature computation uses GPU
# - XGBoost with GPU acceleration
# - CatBoost with GPU acceleration
# ---------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb


from tqdm import tqdm

# -----------------------------
# Dataset
# -----------------------------
class AuthorshipDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# -----------------------------
# Neural Network
# -----------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# NN training
# -----------------------------
def train_neural_network(model, train_loader, test_loader=None, epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch+1) % 1 == 0:
            if test_loader is not None:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for features, labels in test_loader:
                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features).squeeze()
                        predicted = (outputs >= 0.5).float()
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                print(f"Epoch {epoch+1}/{epochs} - Loss {avg_loss:.4f} - TestAcc {correct/total:.4f}")

    return model


# -----------------------------
# Feature preparation (GPU)
# -----------------------------
def prepare_features(df, device="cpu"):
    vec1 = np.stack(df['text1_vec'].values)
    vec2 = np.stack(df['text2_vec'].values)

    t1 = torch.tensor(vec1, dtype=torch.float32, device=device)
    t2 = torch.tensor(vec2, dtype=torch.float32, device=device)

    concat_features = torch.cat([t1, t2], dim=1)
    diff_features = t1 - t2
    prod_features = t1 * t2
    abs_diff_features = torch.abs(t1 - t2)
    cos_sim = (torch.sum(t1*t2, dim=1) / (torch.norm(t1,dim=1)*torch.norm(t2,dim=1)+1e-8)).view(-1,1)

    combined = torch.cat([
        concat_features,
        diff_features,
        prod_features,
        abs_diff_features,
        cos_sim
    ], dim=1)

    return {
        'combined': combined.cpu().numpy(),
        'vec1': t1.cpu().numpy(),
        'vec2': t2.cpu().numpy()
    }


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


# -----------------------------
# Main Training Function
# -----------------------------
def train_authorship_classifiers(train_df, test_df, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    y_train = train_df['same'].values
    y_test = test_df['same'].values

    print("Preparing features on GPU…")
    train_features = prepare_features(train_df, device)
    test_features = prepare_features(test_df, device)

    X_train = train_features['combined']
    X_test = test_features['combined']

    models = {}
    results = []
    # -----------------------------
    # LightGBM GPU
    # -----------------------------
    print("=== Training LightGBM (GPU) ===")

    lgb_model = lgb.LGBMClassifier(
        device='gpu',
        boosting_type='gbdt',
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.9,
        learning_rate=0.05,
        max_depth=-1,  # no max depth
        reg_alpha=0.0,
        reg_lambda=0.0
    )

    lgb_model.fit(X_train, y_train)

    pred_train = lgb_model.predict(X_train)
    pred_test = lgb_model.predict(X_test)

    metrics_train = evaluate_model(y_train, pred_train)
    metrics_test = evaluate_model(y_test, pred_test)

    models['lightgbm'] = lgb_model
    results.append({
        'model': 'LightGBM',
        'train_acc': metrics_train['accuracy'],
        'test_acc': metrics_test['accuracy'],
        'train_f1': metrics_train['f1'],
        'test_f1': metrics_test['f1'],
        "pred_test": pred_test
    })

    # -----------------------------
    # XGBoost GPU
    # -----------------------------
    print("=== Training XGBoost (GPU) ===")
    xgb_model = xgb.XGBClassifier(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        max_depth=8,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    pred_train = xgb_model.predict(X_train)
    pred_test = xgb_model.predict(X_test)

    metrics_train = evaluate_model(y_train, pred_train)
    metrics_test = evaluate_model(y_test, pred_test)

    models['xgboost'] = xgb_model
    results.append({
        'model': 'XGBoost',
        'train_acc': metrics_train['accuracy'],
        'test_acc': metrics_test['accuracy'],
        'train_f1': metrics_train['f1'],
        'test_f1': metrics_test['f1'],
        "pred_test": pred_test
    })

    # -----------------------------
    # CatBoost GPU
    # -----------------------------
    print("=== Training CatBoost (GPU) ===")
    cat_model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        task_type='GPU',
        verbose=False
    )
    cat_model.fit(X_train, y_train)

    pred_train = cat_model.predict(X_train).astype(int)
    pred_test = cat_model.predict(X_test).astype(int)

    metrics_train = evaluate_model(y_train, pred_train)
    metrics_test = evaluate_model(y_test, pred_test)

    models['catboost'] = cat_model
    results.append({
        'model': 'CatBoost',
        'train_acc': metrics_train['accuracy'],
        'test_acc': metrics_test['accuracy'],
        'train_f1': metrics_train['f1'],
        'test_f1': metrics_test['f1'],
        "pred_test": pred_test
    })


    # -----------------------------
    # Neural Network GPU
    # -----------------------------
    print("=== Training Neural Network (GPU) ===")
    scaler_nn = StandardScaler()
    X_train_nn = scaler_nn.fit_transform(X_train)
    X_test_nn = scaler_nn.transform(X_test)

    train_loader = DataLoader(AuthorshipDataset(X_train_nn, y_train), batch_size=1024, shuffle=True)
    test_loader = DataLoader(AuthorshipDataset(X_test_nn, y_test), batch_size=1024, shuffle=False)

    nn_model = SimpleNN(X_train_nn.shape[1])
    nn_model = train_neural_network(nn_model, train_loader, test_loader, epochs=10, device=device)

    nn_model.eval()
    with torch.no_grad():
        p_train = nn_model(torch.FloatTensor(X_train_nn).to(device)).cpu().numpy().squeeze()
        p_test = nn_model(torch.FloatTensor(X_test_nn).to(device)).cpu().numpy().squeeze()

    pred_train = (p_train >= 0.5).astype(int)
    pred_test = (p_test >= 0.5).astype(int)

    metrics_train = evaluate_model(y_train, pred_train)
    metrics_test = evaluate_model(y_test, pred_test)

    models['simple_nn'] = nn_model
    results.append({
        'model': 'SimpleNN',
        'train_acc': metrics_train['accuracy'],
        'test_acc': metrics_test['accuracy'],
        'train_f1': metrics_train['f1'],
        'test_f1': metrics_test['f1'],
        "pred_test": pred_test
    })

    return models, pd.DataFrame(results)

# ==============================
# Ensemble Majority Voting Model
# ==============================
from sklearn.metrics import accuracy_score, f1_score


class MajorityVotingEnsemble:
    def __init__(self, model_names):
        self.model_names = model_names # 예: ['pytorch_nn', 'xgboost', 'catboost']


    def predict(self, df_pred):
        pred_cols = [f"pred_{m}" for m in self.model_names]
        return df_pred[pred_cols].mode(axis=1)[0]


    def evaluate(self, df_pred, true_label_col="true_label"):
        y_true = df_pred[true_label_col]
        y_pred = self.predict(df_pred)


        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)


        print("\n==============================")
        print("🔮 Majority Voting Ensemble 결과")
        print("==============================")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("==============================\n")


        return {"accuracy": acc, "f1": f1}