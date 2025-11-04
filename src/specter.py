from base_vectorizer import BaseLoadVectorizer
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class SpecterVectorizer(BaseLoadVectorizer):
    def __init__(self, model_name='allenai/specter', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self
    
    def embed(self, dataset):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        text1_vectors = []
        text2_vectors = []
        
        with torch.no_grad():
            for text in tqdm(dataset['text1']):
                inputs = self.tokenizer(text, padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                text1_vectors.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0])
            
            for text in tqdm(dataset['text2']):
                inputs = self.tokenizer(text, padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                text2_vectors.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0])
        
        df_data = []
        for i in range(len(dataset)):
            df_data.append({
                'text1': dataset['text1'][i],
                'text2': dataset['text2'][i],
                'text1_vec': text1_vectors[i],
                'text2_vec': text2_vectors[i],
                'same': dataset['same'][i]
            })
        
        return pd.DataFrame(df_data)