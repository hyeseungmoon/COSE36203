from base_vectorizer import BaseVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class TfIdfVectorizer(BaseVectorizer):
    
    def __init__(self, ngram_range=(1, 2), max_features=1000, min_df=5, max_df=0.8):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.is_trained = False
    
    def train(self, dataset):
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df
        )
        
        all_texts = []
        
        all_texts.extend(dataset['text1'])
        all_texts.extend(dataset['text2'])
        
        print(f"Vectorizer 학습 중...")
        self.vectorizer.fit(all_texts)
        self.is_trained = True
        
        return self
    
    def embed(self, dataset):
       
        if not self.is_trained:
            raise ValueError("Vectorizer가 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        text1_vectors = self.vectorizer.transform(dataset['text1']).toarray()
        text2_vectors = self.vectorizer.transform(dataset['text2']).toarray()
        
        df_data = []
        for i in range(len(dataset)):
            df_data.append({
                'text1': dataset['text1'][i],
                'text2': dataset['text2'][i],
                'text1_vec': text1_vectors[i],
                'text2_vec': text2_vectors[i],
                'same': dataset['same'][i]
            })
        
        df = pd.DataFrame(df_data)
        
        return df
    
    def save(self, file_name):
        
        if not self.is_trained:
            raise ValueError("저장할 학습된 모델이 없습니다. train()을 먼저 호출하세요.")
        
        if not file_name.endswith('.pkl'):
            file_name += '.pkl'
        
        save_data = {
            'vectorizer': self.vectorizer,
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df
        }
        
        # pickle로 저장
        with open(file_name, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Vectorizer가 '{file_name}'에 저장되었습니다.")
    
    def load(self, file_name):
        if not file_name.endswith('.pkl'):
            file_name += '.pkl'
        
        with open(file_name, 'rb') as f:
            save_data = pickle.load(f)
        
        self.vectorizer = save_data['vectorizer']
        self.ngram_range = save_data['ngram_range']
        self.max_features = save_data['max_features']
        self.min_df = save_data['min_df']
        self.max_df = save_data['max_df']
        self.is_trained = True
        print(f"Vectorizer가 '{file_name}'에서 로드되었습니다.")
        
        return self
    
