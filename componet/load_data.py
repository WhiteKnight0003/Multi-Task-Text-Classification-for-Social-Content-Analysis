import pandas as pd

class DataLoader:

    def __init__(self, emotion_path, violence_path, hate_path):
        self.emotion_path = emotion_path
        self.violence_path = violence_path
        self.hate_path = hate_path
    
    def load_data(self):
        emotion_df = pd.read_csv(self.emotion_path)
        violence_df = pd.read_csv(self.violence_path)
        hate_df = pd.read_csv(self.hate_path) 
        return emotion_df, violence_df, hate_df
    
    
if __name__ == "__main__":
    # Define paths
    emotion_path = './data/Emotions/text.csv'
    violence_path = './data/Gender-Based_Violence_Tweet_Classification/train.csv'
    hate_path = './data/Hate_Speech_and_Offensive_Language_Dataset/labeled_data.csv'

    data = DataLoader(emotion_path, violence_path,  hate_path)
    emotion_df, violence_df, hate_df = data.load_data()
    
    print(emotion_df['label'].unique())
    print(violence_df['type'].unique())
    print(hate_df['class'].unique())