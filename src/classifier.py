import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


class AuthorshipDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class SiameseNN(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mid = x.shape[1] // 2
        vec1 = x[:, :mid]
        vec2 = x[:, mid:]

        enc1 = self.encoder(vec1)
        enc2 = self.encoder(vec2)

        combined = torch.cat([enc1, enc2], dim=1)
        return self.classifier(combined)


def train_neural_network(model, train_loader, test_loader=None, epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5)

    for epoch in range(epochs):
        # Training
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

        # Evaluation
        if (epoch + 1) % 10 == 0:
            if test_loader is not None:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for features, labels in test_loader:
                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features).squeeze()
                        predicted = (outputs >= 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                test_acc = correct / total
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model


def prepare_features(df):
    vec1 = np.stack(df['text1_vec'].values)
    vec2 = np.stack(df['text2_vec'].values)

    concat_features = np.concatenate([vec1, vec2], axis=1)
    diff_features = vec1 - vec2
    prod_features = vec1 * vec2
    abs_diff_features = np.abs(vec1 - vec2)

    cos_sim = np.sum(vec1 * vec2, axis=1) / (
            np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1) + 1e-8
    )
    cos_sim = cos_sim.reshape(-1, 1)

    combined_features = np.concatenate([
        concat_features, diff_features, prod_features,
        abs_diff_features, cos_sim
    ], axis=1)

    return {
        'concat': concat_features,
        'combined': combined_features,
        'vec1': vec1,
        'vec2': vec2
    }


def evaluate_model(y_true, y_pred):
    """평가 지표 계산"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


def train_authorship_classifiers(train_df, test_df, use_gpu=False):
    print("=" * 60)
    print("저자 동일성 분류 모델 학습 시작")
    print("=" * 60)

    print(f"\n학습 데이터 크기: {len(train_df)} samples")
    print(f"테스트 데이터 크기: {len(test_df)} samples")

    y_train = train_df['same'].values
    y_test = test_df['same'].values
    print(f"\n[Train] 클래스 분포 - 같은 저자: {np.sum(y_train)}, 다른 저자: {len(y_train) - np.sum(y_train)}")
    print(f"[Test] 클래스 분포 - 같은 저자: {np.sum(y_test)}, 다른 저자: {len(y_test) - np.sum(y_test)}")

    # Feature 준비
    print("\nFeature 준비 중...")
    train_features = prepare_features(train_df)
    test_features = prepare_features(test_df)

    # 모델 저장용 딕셔너리
    models = {}
    results = []

    # GPU 설정
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")

    # ========================================
    # 1. 코사인 유사도 기반
    # ========================================
    print("\n" + "=" * 60)
    print("1. 코사인 유사도 분류기 학습")
    print("=" * 60)

    train_cos_sim = cosine_similarity(train_features['vec1'], train_features['vec2']).diagonal()
    test_cos_sim = cosine_similarity(test_features['vec1'], test_features['vec2']).diagonal()

    best_threshold = 0.5
    best_acc = 0
    for threshold in np.linspace(0, 1, 101):
        pred = (train_cos_sim >= threshold).astype(int)
        acc = np.mean(pred == y_train)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    train_pred = (train_cos_sim >= best_threshold).astype(int)
    test_pred = (test_cos_sim >= best_threshold).astype(int)

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    models['cosine_similarity'] = {
        'threshold': best_threshold,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    print(f"최적 Threshold: {best_threshold:.4f}")
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")

    results.append({
        'model': 'Cosine Similarity',
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1']
    })

    # ========================================
    # 2. SVM
    # ========================================
    print("\n" + "=" * 60)
    print("2. SVM 분류기 학습")
    print("=" * 60)

    scaler_svm = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(train_features['combined'])
    X_test_svm = scaler_svm.transform(test_features['combined'])

    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train_svm, y_train)

    train_pred = svm_model.predict(X_train_svm)
    test_pred = svm_model.predict(X_test_svm)

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    models['svm'] = {
        'model': svm_model,
        'scaler': scaler_svm,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")

    results.append({
        'model': 'SVM',
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1']
    })

    # ========================================
    # 3. Random Forest
    # ========================================
    print("\n" + "=" * 60)
    print("3. Random Forest 분류기 학습")
    print("=" * 60)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                      min_samples_split=10, random_state=42,
                                      n_jobs=-1)
    rf_model.fit(train_features['combined'], y_train)

    train_pred = rf_model.predict(train_features['combined'])
    test_pred = rf_model.predict(test_features['combined'])

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    models['random_forest'] = {
        'model': rf_model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")

    results.append({
        'model': 'Random Forest',
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1']
    })

    # ========================================
    # 4. Logistic Regression
    # ========================================
    print("\n" + "=" * 60)
    print("4. Logistic Regression 분류기 학습")
    print("=" * 60)

    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(train_features['combined'])
    X_test_lr = scaler_lr.transform(test_features['combined'])

    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_model.fit(X_train_lr, y_train)

    train_pred = lr_model.predict(X_train_lr)
    test_pred = lr_model.predict(X_test_lr)

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    models['logistic_regression'] = {
        'model': lr_model,
        'scaler': scaler_lr,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")

    results.append({
        'model': 'Logistic Regression',
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1']
    })

    # ========================================
    # 5. Naive Bayes
    # ========================================
    print("\n" + "=" * 60)
    print("5. Gaussian Naive Bayes 분류기 학습")
    print("=" * 60)

    nb_model = GaussianNB()
    nb_model.fit(train_features['combined'], y_train)

    train_pred = nb_model.predict(train_features['combined'])
    test_pred = nb_model.predict(test_features['combined'])

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    models['naive_bayes'] = {
        'model': nb_model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")

    results.append({
        'model': 'Naive Bayes',
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1']
    })

    # ========================================
    # 6. Simple Neural Network
    # ========================================
    print("\n" + "=" * 60)
    print("6. Simple Neural Network 학습")
    print("=" * 60)

    scaler_nn = StandardScaler()
    X_train_nn = scaler_nn.fit_transform(train_features['combined'])
    X_test_nn = scaler_nn.transform(test_features['combined'])

    dataset_train = AuthorshipDataset(X_train_nn, y_train)
    dataset_test = AuthorshipDataset(X_test_nn, y_test)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    simple_nn = SimpleNN(X_train_nn.shape[1])
    simple_nn = train_neural_network(simple_nn, train_loader, test_loader, epochs=50,
                                     lr=0.001, device=device)

    # 평가
    simple_nn.eval()
    with torch.no_grad():
        train_pred_prob = simple_nn(torch.FloatTensor(X_train_nn).to(device)).squeeze().cpu().numpy()
        test_pred_prob = simple_nn(torch.FloatTensor(X_test_nn).to(device)).squeeze().cpu().numpy()

    train_pred = (train_pred_prob >= 0.5).astype(int)
    test_pred = (test_pred_prob >= 0.5).astype(int)

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    models['simple_nn'] = {
        'model': simple_nn.cpu(),
        'scaler': scaler_nn,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")

    results.append({
        'model': 'Simple NN',
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1']
    })

    # ========================================
    # 7. Deep Neural Network
    # ========================================
    print("\n" + "=" * 60)
    print("7. Deep Neural Network 학습")
    print("=" * 60)

    deep_nn = DeepNN(X_train_nn.shape[1])
    deep_nn = train_neural_network(deep_nn, train_loader, test_loader, epochs=50,
                                   lr=0.0005, device=device)

    # 평가
    deep_nn.eval()
    with torch.no_grad():
        train_pred_prob = deep_nn(torch.FloatTensor(X_train_nn).to(device)).squeeze().cpu().numpy()
        test_pred_prob = deep_nn(torch.FloatTensor(X_test_nn).to(device)).squeeze().cpu().numpy()

    train_pred = (train_pred_prob >= 0.5).astype(int)
    test_pred = (test_pred_prob >= 0.5).astype(int)

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    models['deep_nn'] = {
        'model': deep_nn.cpu(),
        'scaler': scaler_nn,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")

    results.append({
        'model': 'Deep NN',
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1']
    })

    # ========================================
    # 8. Siamese Neural Network
    # ========================================
    print("\n" + "=" * 60)
    print("8. Siamese Neural Network 학습")
    print("=" * 60)

    scaler_siamese = StandardScaler()
    X_train_siamese = scaler_siamese.fit_transform(train_features['concat'])
    X_test_siamese = scaler_siamese.transform(test_features['concat'])

    dataset_train_siamese = AuthorshipDataset(X_train_siamese, y_train)
    dataset_test_siamese = AuthorshipDataset(X_test_siamese, y_test)
    train_loader_siamese = DataLoader(dataset_train_siamese, batch_size=32, shuffle=True)
    test_loader_siamese = DataLoader(dataset_test_siamese, batch_size=32, shuffle=False)

    vec_dim = train_features['vec1'].shape[1]
    siamese_nn = SiameseNN(vec_dim)
    siamese_nn = train_neural_network(siamese_nn, train_loader_siamese, test_loader_siamese,
                                      epochs=50, lr=0.001, device=device)

    # 평가
    siamese_nn.eval()
    with torch.no_grad():
        train_pred_prob = siamese_nn(torch.FloatTensor(X_train_siamese).to(device)).squeeze().cpu().numpy()
        test_pred_prob = siamese_nn(torch.FloatTensor(X_test_siamese).to(device)).squeeze().cpu().numpy()

    train_pred = (train_pred_prob >= 0.5).astype(int)
    test_pred = (test_pred_prob >= 0.5).astype(int)

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    models['siamese_nn'] = {
        'model': siamese_nn.cpu(),
        'scaler': scaler_siamese,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")

    results.append({
        'model': 'Siamese NN',
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1']
    })

    # ========================================
    # 학습 완료 요약
    # ========================================
    print("\n" + "=" * 60)
    print("전체 학습 완료!")
    print("=" * 60)

    # 결과 테이블 출력
    results_df = pd.DataFrame(results)
    print("\n📊 모델 성능 비교:")
    print("-" * 80)
    print(f"{'Model':<20} {'Train Acc':>10} {'Test Acc':>10} {'Train F1':>10} {'Test F1':>10}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['model']:<20} {row['train_acc']:>10.4f} {row['test_acc']:>10.4f} "
              f"{row['train_f1']:>10.4f} {row['test_f1']:>10.4f}")
    print("-" * 80)

    # 최고 성능 모델
    best_model = results_df.loc[results_df['test_acc'].idxmax()]
    print(f"\n🏆 최고 성능 모델: {best_model['model']} (Test Acc: {best_model['test_acc']:.4f})")

    # ========================================
    # 전체 예측 결과 DataFrame 생성
    # ========================================
    print("\n" + "=" * 60)
    print("예측 결과 DataFrame 생성 중...")
    print("=" * 60)

    # Train 예측
    train_predictions_df = train_df.copy()
    train_predictions_df['true_label'] = y_train

    # 1. Cosine Similarity
    train_cos_sim = cosine_similarity(train_features['vec1'], train_features['vec2']).diagonal()
    train_predictions_df['pred_cosine'] = (train_cos_sim >= models['cosine_similarity']['threshold']).astype(int)

    # 2. SVM
    X_train_svm = models['svm']['scaler'].transform(train_features['combined'])
    train_predictions_df['pred_svm'] = models['svm']['model'].predict(X_train_svm)

    # 3. Random Forest
    train_predictions_df['pred_rf'] = models['random_forest']['model'].predict(train_features['combined'])

    # 4. Logistic Regression
    X_train_lr = models['logistic_regression']['scaler'].transform(train_features['combined'])
    train_predictions_df['pred_lr'] = models['logistic_regression']['model'].predict(X_train_lr)

    # 5. Naive Bayes
    train_predictions_df['pred_nb'] = models['naive_bayes']['model'].predict(train_features['combined'])

    # 6. Simple NN
    X_train_nn = models['simple_nn']['scaler'].transform(train_features['combined'])
    models['simple_nn']['model'].eval()
    with torch.no_grad():
        pred_simple_nn = models['simple_nn']['model'](torch.FloatTensor(X_train_nn)).squeeze().numpy()
    train_predictions_df['pred_simple_nn'] = (pred_simple_nn >= 0.5).astype(int)

    # 7. Deep NN
    models['deep_nn']['model'].eval()
    with torch.no_grad():
        pred_deep_nn = models['deep_nn']['model'](torch.FloatTensor(X_train_nn)).squeeze().numpy()
    train_predictions_df['pred_deep_nn'] = (pred_deep_nn >= 0.5).astype(int)

    # 8. Siamese NN
    X_train_siamese = models['siamese_nn']['scaler'].transform(train_features['concat'])
    models['siamese_nn']['model'].eval()
    with torch.no_grad():
        pred_siamese = models['siamese_nn']['model'](torch.FloatTensor(X_train_siamese)).squeeze().numpy()
    train_predictions_df['pred_siamese_nn'] = (pred_siamese >= 0.5).astype(int)

    # Test 예측
    test_predictions_df = test_df.copy()
    test_predictions_df['true_label'] = y_test

    # 1. Cosine Similarity
    test_cos_sim = cosine_similarity(test_features['vec1'], test_features['vec2']).diagonal()
    test_predictions_df['pred_cosine'] = (test_cos_sim >= models['cosine_similarity']['threshold']).astype(int)

    # 2. SVM
    X_test_svm = models['svm']['scaler'].transform(test_features['combined'])
    test_predictions_df['pred_svm'] = models['svm']['model'].predict(X_test_svm)

    # 3. Random Forest
    test_predictions_df['pred_rf'] = models['random_forest']['model'].predict(test_features['combined'])

    # 4. Logistic Regression
    X_test_lr = models['logistic_regression']['scaler'].transform(test_features['combined'])
    test_predictions_df['pred_lr'] = models['logistic_regression']['model'].predict(X_test_lr)

    # 5. Naive Bayes
    test_predictions_df['pred_nb'] = models['naive_bayes']['model'].predict(test_features['combined'])

    # 6. Simple NN
    X_test_nn = models['simple_nn']['scaler'].transform(test_features['combined'])
    models['simple_nn']['model'].eval()
    with torch.no_grad():
        pred_simple_nn = models['simple_nn']['model'](torch.FloatTensor(X_test_nn)).squeeze().numpy()
    test_predictions_df['pred_simple_nn'] = (pred_simple_nn >= 0.5).astype(int)

    # 7. Deep NN
    models['deep_nn']['model'].eval()
    with torch.no_grad():
        pred_deep_nn = models['deep_nn']['model'](torch.FloatTensor(X_test_nn)).squeeze().numpy()
    test_predictions_df['pred_deep_nn'] = (pred_deep_nn >= 0.5).astype(int)

    # 8. Siamese NN
    X_test_siamese = models['siamese_nn']['scaler'].transform(test_features['concat'])
    models['siamese_nn']['model'].eval()
    with torch.no_grad():
        pred_siamese = models['siamese_nn']['model'](torch.FloatTensor(X_test_siamese)).squeeze().numpy()
    test_predictions_df['pred_siamese_nn'] = (pred_siamese >= 0.5).astype(int)

    print(f"\n✓ Train 예측 DataFrame 생성 완료: {train_predictions_df.shape}")
    print(f"✓ Test 예측 DataFrame 생성 완료: {test_predictions_df.shape}")
    print(f"\n예측 컬럼: {[col for col in train_predictions_df.columns if col.startswith('pred_')]}")

    return models, results_df, train_predictions_df, test_predictions_df


if __name__ == "__main__":
    # 사용 예시
    print("예시 데이터로 테스트...")

    n_train = 1000
    n_test = 200
    vec_dim = 128

    np.random.seed(42)
    train_df = pd.DataFrame({
        'text1_vec': [np.random.randn(vec_dim) for _ in range(n_train)],
        'text2_vec': [np.random.randn(vec_dim) for _ in range(n_train)],
        'same': np.random.randint(0, 2, n_train)
    })

    test_df = pd.DataFrame({
        'text1_vec': [np.random.randn(vec_dim) for _ in range(n_test)],
        'text2_vec': [np.random.randn(vec_dim) for _ in range(n_test)],
        'same': np.random.randint(0, 2, n_test)
    })

    # 모델 학습
    trained_models, results, train_preds, test_preds = train_authorship_classifiers(
        train_df, test_df, use_gpu=False
    )

    print("\n✅ 학습 완료!")
    print(f"학습된 모델 개수: {len(trained_models)}")
    print(f"결과 DataFrame shape: {results.shape}")
    print(f"Train 예측 DataFrame shape: {train_preds.shape}")
    print(f"Test 예측 DataFrame shape: {test_preds.shape}")

    # 예측 결과 확인
    print("\n[Train] 처음 5개 샘플의 예측 결과:")
    pred_cols = [col for col in train_preds.columns if col.startswith('pred_')]
    print(train_preds[['true_label'] + pred_cols].head())