from abc import ABC, abstractmethod

class BaseVectorizer(ABC):
    """추상 기본 클래스"""
    
    @abstractmethod
    def train(self, dataset):
        """
            해당 dataset으로 학습
            dataset : Dataset
        """
        pass
    
    @abstractmethod
    def embed(self, dataset):
        """
            해당 dataset의 텍스트를 벡터로 변환
            dataset : Dataset
            return : {
                'text1': str,
                'text2': str,
                'text1_vec': np.ndarray,
                'text2_vec': np.ndarray,
                'same': int
            } 형태의 DataFrame
        """
        pass
    
    @abstractmethod
    def save(self, file_name):
        """
            저장된 모델 파일로 저장
            file_name : str
        """
        pass
    
    @abstractmethod
    def load(self, file_name):
        """
            저장된 모델 로드
            file_name : str
        """
        pass


class BaseLoadVectorizer(ABC):
    
    @abstractmethod
    def load(self):
        """"허깅페이스에서 모델 로드"""
        pass
    
    @abstractmethod
    def embed(self, dataset):
        """
        해당 dataset의 텍스트를 벡터로 변환
            dataset : Dataset
            return : {
                'text1': str,
                'text2': str,
                'text1_vec': np.ndarray,
                'text2_vec': np.ndarray,
                'same': int
            } 형태의 DataFrame
        """
        pass