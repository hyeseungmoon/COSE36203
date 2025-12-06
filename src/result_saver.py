import pickle
import torch
from pathlib import Path


def save_results(models, results, train_preds_df, test_preds_df, save_dir='./results', prefix=''):
    """학습 결과 저장
    
    Args:
        models: 학습된 모델 딕셔너리
        results: 결과 DataFrame
        train_preds_df: 학습 예측 DataFrame
        test_preds_df: 테스트 예측 DataFrame
        save_dir: 저장 디렉토리
        prefix: 파일명 접두사 (예: 'specter', 'bert')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # prefix 처리
    prefix = f"{prefix}_" if prefix else ""
    
    # 1. DataFrame들은 pickle로 바로 저장
    with open(save_dir / f'{prefix}results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(save_dir / f'{prefix}train_preds_df.pkl', 'wb') as f:
        pickle.dump(train_preds_df, f)
    with open(save_dir / f'{prefix}test_preds_df.pkl', 'wb') as f:
        pickle.dump(test_preds_df, f)
    
    # 2. Models 저장 (sklearn과 PyTorch 분리)
    sklearn_models = {}
    for name, model_dict in models.items():
        if name == 'simple_nn':
            # PyTorch 모델은 state_dict로 저장
            torch.save({
                'model_state_dict': model_dict['model'].state_dict(),
                'scaler': model_dict['scaler'],
                'train_metrics': model_dict['train_metrics'],
                'test_metrics': model_dict['test_metrics'],
                'input_dim': list(model_dict['model'].parameters())[0].shape[1]
            }, save_dir / f'{prefix}simple_nn.pt')
        else:
            # sklearn 모델은 그대로 저장
            sklearn_models[name] = model_dict
    
    with open(save_dir / f'{prefix}sklearn_models.pkl', 'wb') as f:
        pickle.dump(sklearn_models, f)
    
    print(f"✅ 결과 저장 완료: {save_dir} (prefix: '{prefix[:-1] if prefix else 'none'}')")


def load_results(save_dir='./results', prefix=''):
    """학습 결과 로드
    
    Args:
        save_dir: 저장 디렉토리
        prefix: 파일명 접두사 (예: 'specter', 'bert')
    
    Returns:
        models, results, train_preds_df, test_preds_df
    """
    from classifier import SimpleNN  # 모델 클래스 import 필요
    
    save_dir = Path(save_dir)
    
    # prefix 처리
    prefix = f"{prefix}_" if prefix else ""
    
    # 1. DataFrame 로드
    with open(save_dir / f'{prefix}results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open(save_dir / f'{prefix}train_preds_df.pkl', 'rb') as f:
        train_preds_df = pickle.load(f)
    with open(save_dir / f'{prefix}test_preds_df.pkl', 'rb') as f:
        test_preds_df = pickle.load(f)
    
    # 2. sklearn 모델 로드
    with open(save_dir / f'{prefix}sklearn_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    # 3. PyTorch 모델 로드
    checkpoint = torch.load(save_dir / f'{prefix}simple_nn.pt', weights_only=False)
    simple_nn = SimpleNN(checkpoint['input_dim'])
    simple_nn.load_state_dict(checkpoint['model_state_dict'])
    simple_nn.eval()
    
    models['simple_nn'] = {
        'model': simple_nn,
        'scaler': checkpoint['scaler'],
        'train_metrics': checkpoint['train_metrics'],
        'test_metrics': checkpoint['test_metrics']
    }
    
    print(f"✅ 결과 로드 완료: {save_dir} (prefix: '{prefix[:-1] if prefix else 'none'}')")
    return models, results, train_preds_df, test_preds_df