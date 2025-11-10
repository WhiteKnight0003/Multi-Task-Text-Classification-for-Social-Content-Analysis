from sklearn.model_selection import train_test_split
from componet.load_data import DataLoader
from componet.preprocessor import DataPreprocessor

class DataSplitter:
    """Split data into train and test sets"""
    
    def __init__(self, test_size=0.2, random_state=100):
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, emotion_df, violence_df, hate_df):
        """Split all datasets"""
        # Emotion split
        x_emotion = emotion_df['text']
        y_emotion = emotion_df['label']
        x_emotion_train, x_emotion_test, y_emotion_train, y_emotion_test = train_test_split(
            x_emotion, y_emotion, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y_emotion
        )     


        # Violence split
        x_violence = violence_df['text']
        y_violence = violence_df['label']
        x_violence_train, x_violence_test, y_violence_train, y_violence_test = train_test_split(
            x_violence, y_violence,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_violence
        )
        
        # Hate split
        x_hate = hate_df['text']
        y_hate = hate_df['label']
        x_hate_train, x_hate_test, y_hate_train, y_hate_test = train_test_split(
            x_hate, y_hate,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_hate
        ) 
        
        return {
            'emotion': (x_emotion_train, x_emotion_test, y_emotion_train, y_emotion_test),
            'violence': (x_violence_train, x_violence_test, y_violence_train, y_violence_test),
            'hate': (x_hate_train, x_hate_test, y_hate_train, y_hate_test)
        }
    
if __name__ == "__main__":
    # Define paths
    emotion_path = './data/Emotions/text.csv'
    violence_path = './data/Gender-Based_Violence_Tweet_Classification/train.csv'
    hate_path = './data/Hate_Speech_and_Offensive_Language_Dataset/labeled_data.csv'

    data = DataLoader(emotion_path, violence_path, hate_path)
    emotion_df, violence_df, hate_df = data.load_data()

    preprocessor = DataPreprocessor()
    emotion_df, violence_df, hate_df = preprocessor.process(
        emotion_df, violence_df, hate_df
    )
    
    splitter = DataSplitter()
    train_test_splits = splitter.split(emotion_df, violence_df, hate_df)
    print(train_test_splits)