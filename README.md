# Authorship Verification with Machine Learning

A comprehensive machine learning project for authorship verification using multiple text embedding techniques and classification models.

## Overview

This project implements an authorship verification system that determines whether two text samples are written by the same author. It uses state-of-the-art text embedding models (BERT and SPECTER) combined with various machine learning classifiers to achieve robust authorship verification.

## Features

- **Multiple Text Embedding Methods**:
  - BERT (bert-base-uncased)
  - SPECTER (allenai/specter) - Scientific paper embeddings

- **8 Classification Models**:
  - Cosine Similarity baseline
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - Gaussian Naive Bayes
  - Simple Neural Network
  - Deep Neural Network
  - Siamese Neural Network

- **Comprehensive Evaluation Framework**:
  - Multiple metrics (Accuracy, Precision, Recall, F1-Score, MCC)
  - Confusion matrix analysis
  - Model comparison and ensemble predictions
  - Disagreement analysis between models

## Project Structure

```
src/
├── base_vectorizer.py      # Abstract base classes for vectorizers
├── bert.py                  # BERT-based text vectorizer
├── specter.py              # SPECTER-based text vectorizer
├── classifier.py           # 8 classification models implementation
├── main.ipynb              # Main execution notebook
└── embeddings/             # Pre-computed embeddings storage
    ├── train_bert.pkl
    ├── test_bert.pkl
    ├── train_specter.pkl
    └── test_specter.pkl
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional, for faster training)

### Required Packages

```bash
pip install torch transformers datasets
pip install scikit-learn pandas numpy tqdm
pip install matplotlib seaborn
```

## Usage

### 1. Load Dataset and Generate Embeddings

```python
from datasets import load_dataset
from bert import BertVectorizer
from specter import SpecterVectorizer

# Load authorship verification dataset
train_dataset = load_dataset("swan07/authorship-verification", split="train[:5000]")
test_dataset = load_dataset("swan07/authorship-verification", split="test[:1000]")

# Generate BERT embeddings
bert_vectorizer = BertVectorizer()
bert_vectorizer.load()
train_bert = bert_vectorizer.embed(train_dataset)
test_bert = bert_vectorizer.embed(test_dataset)

# Generate SPECTER embeddings
specter_vectorizer = SpecterVectorizer()
specter_vectorizer.load()
train_specter = specter_vectorizer.embed(train_dataset)
test_specter = specter_vectorizer.embed(test_dataset)
```

### 2. Train Classification Models

```python
from classifier import train_authorship_classifiers

# Train all 8 models
models, results, train_preds_df, test_preds_df = train_authorship_classifiers(
    train_df=train_data,
    test_df=test_data,
    use_gpu=True  # Set to False if no GPU available
)
```

### 3. Evaluate and Compare Models

The training process automatically provides:
- Individual model performance metrics
- Confusion matrices for each model
- Best performing model identification
- Comprehensive prediction DataFrames

### 4. Use the Evaluation Framework (from main.ipynb)

```python
from main import MultiModelEvaluator, evaluate_predictions

# Initialize evaluator
evaluator = MultiModelEvaluator()

# Print detailed evaluation report
evaluator.print_report(test_preds_df)

# Compare models by specific metric
f1_comparison = evaluator.compare_models(test_preds_df, metric='f1_score')

# Generate ensemble predictions
ensemble = evaluator.get_ensemble_prediction(test_preds_df, method='majority')

# Analyze disagreements between models
disagreements = evaluator.analyze_disagreements(test_preds_df)
```

## Model Performance

Based on experiments with 5,000 training samples and 1,000 test samples using SPECTER embeddings:

| Model | Train Acc | Test Acc | Train F1 | Test F1 |
|-------|-----------|----------|----------|---------|
| SVM | 0.9036 | **0.6380** | 0.8980 | 0.6065 |
| Simple NN | 1.0000 | 0.6360 | 1.0000 | 0.6224 |
| Naive Bayes | 0.6238 | 0.6270 | 0.6295 | **0.6368** |
| Deep NN | 0.9998 | 0.6180 | 0.9998 | 0.6180 |
| Random Forest | 1.0000 | 0.6170 | 1.0000 | 0.5413 |
| Cosine Similarity | 0.5960 | 0.6170 | 0.5254 | 0.5402 |
| Siamese NN | 0.9956 | 0.5820 | 0.9955 | 0.6049 |
| Logistic Regression | 0.9984 | 0.5740 | 0.9983 | 0.5680 |

**Best Model**: SVM with 63.8% test accuracy and 60.65% F1-score

**Ensemble Accuracy**: 64.5% (majority voting across all models)

## Architecture Details

### Vectorizers

**BaseVectorizer (Abstract)**:
- `train(dataset)`: Train the vectorizer
- `embed(dataset)`: Convert text to vectors
- `save(file_name)`: Save model
- `load(file_name)`: Load model

**BaseLoadVectorizer (Abstract)**:
- `load()`: Load pre-trained model from HuggingFace
- `embed(dataset)`: Convert text to vectors

### Neural Network Architectures

**SimpleNN**:
- 4-layer network: 256 → 128 → 64 → 1
- ReLU activations with Dropout (0.2-0.3)
- Sigmoid output for binary classification

**DeepNN**:
- 5-layer network: 512 → 256 → 128 → 64 → 1
- Batch normalization after each layer
- Higher dropout rates (0.2-0.4)

**SiameseNN**:
- Twin encoder networks sharing weights
- Processes both text embeddings separately
- Combines encoded representations for classification
- Encoder: 256 → 128 → 64
- Classifier: 64 → 1

### Feature Engineering

The classifier module creates multiple feature representations:
- **Concatenation**: [vec1, vec2]
- **Difference**: vec1 - vec2
- **Element-wise product**: vec1 * vec2
- **Absolute difference**: |vec1 - vec2|
- **Cosine similarity**: cos_sim(vec1, vec2)

Combined features are used for SVM, Random Forest, Logistic Regression, Naive Bayes, Simple NN, and Deep NN.

## Dataset

The project uses the `swan07/authorship-verification` dataset from HuggingFace, which contains:
- **text1**: First text sample
- **text2**: Second text sample
- **same**: Binary label (1 if same author, 0 if different)

## GPU Support

All models support GPU acceleration when available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Neural networks are automatically moved to GPU if available. Set `use_gpu=True` in `train_authorship_classifiers()` to enable.

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions / Total positive predictions
- **Recall (Sensitivity)**: Correct positive predictions / Total actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient
- **Specificity**: True negative rate
- **Confusion Matrix**: Detailed classification results (TP, TN, FP, FN)

## Advanced Features

### Ensemble Learning
Combine predictions from multiple models:
- **Majority Voting**: Most common prediction across models
- **Unanimous**: All models must agree

### Disagreement Analysis
Identify samples where models disagree and analyze ensemble performance on those difficult cases.

### Model Comparison
Compare models across different metrics to find the best performer for your specific use case.

## Saving and Loading Embeddings

```python
import pickle
from pathlib import Path

# Save embeddings
Path("./embeddings").mkdir(exist_ok=True)

with open('./embeddings/train_specter.pkl', 'wb') as f:
    pickle.dump(train_specter, f)

# Load embeddings
with open('./embeddings/train_specter.pkl', 'rb') as f:
    train_data = pickle.load(f)
```

## Training Configuration

### Neural Network Training
- **Epochs**: 50
- **Learning Rate**: 0.001 (Simple/Siamese), 0.0005 (Deep)
- **Batch Size**: 32
- **Optimizer**: Adam with weight decay (1e-5)
- **Loss**: Binary Cross Entropy (BCE)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

### Traditional ML Models
- **SVM**: RBF kernel, C=1.0, gamma='scale'
- **Random Forest**: 200 trees, max_depth=20, min_samples_split=10
- **Logistic Regression**: max_iter=1000, C=1.0

## Contributing

Contributions are welcome! Areas for improvement:
- Additional embedding methods (RoBERTa, SciBERT, etc.)
- Hyperparameter optimization
- More advanced ensemble techniques
- Cross-validation implementation
- Additional evaluation metrics

## License

This project is for educational and research purposes.

## Acknowledgments

- BERT: [Google Research](https://github.com/google-research/bert)
- SPECTER: [AllenAI](https://github.com/allenai/specter)
- Dataset: [swan07/authorship-verification](https://huggingface.co/datasets/swan07/authorship-verification)