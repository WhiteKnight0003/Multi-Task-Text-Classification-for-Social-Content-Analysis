from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self):
        self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.violence_labels = ['sexual_violence', 'physical_violence', 
                               'emotional_violence','economic_violence', 'harmful_traditional_practice']
        self.hate_labels = ['hate_speech', ' offensive_language','neither']
    
    def plot_confusion_matrix(self, y_true, y_pred, title, labels):
        """Plot confusion matrix"""
        cf = confusion_matrix(y_true, y_pred, normalize='true')
        plt.figure(figsize=(7, 6))
        sns.heatmap(cf, annot=True, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Prediction')
        plt.tight_layout()
    
    def evaluate(self, processed_data, emotion_pred, violence_pred, hate_pred):
        """Evaluate all tasks"""
        self.plot_confusion_matrix(
            processed_data['emotion']['y_test'],
            emotion_pred,
            'Confusion Matrix for Emotion',
            self.emotion_labels
        )
        
        self.plot_confusion_matrix(
            processed_data['violence']['y_test'],
            violence_pred,
            'Confusion Matrix for Violence',
            self.violence_labels
        )
        
        self.plot_confusion_matrix(
            processed_data['hate']['y_test'],
            hate_pred,
            'Confusion Matrix for Hate',
            self.hate_labels
        )
        
        plt.show()


class TextClassifier:
    """End-to-end text classifier"""
    
    def __init__(self, model, tokenizer, preprocessor):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        
        self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.violence_labels = ['sexual_violence', 'physical_violence', 
                               'emotional_violence','economic_violence', 'harmful_traditional_practice']
        self.hate_labels = ['hate_speech', ' offensive_language','neither']
    
    def classify(self, text):
        """Classify input text for all tasks"""
        # Preprocess
        cleaned_text = self.preprocessor.remove_stopwords(text)
        
        # Tokenize and pad
        padded = self.tokenizer.tokenize_and_pad([cleaned_text])
        
        # Predict
        predictions = self.model.predict({
            'emotion_input': padded,
            'violence_input': padded,
            'hate_input': padded
        })
        
        # Get predictions
        emotion_idx = np.argmax(predictions[0], axis=1)[0]
        violence_idx = np.argmax(predictions[1], axis=1)[0]
        hate_idx = np.argmax(predictions[2], axis=1)[0]
        
        # Get probabilities
        emotion_prob = np.max(predictions[0])
        violence_prob = np.max(predictions[1])
        hate_prob = np.max(predictions[2])
        
        results = {
            'Emotion': f"{self.emotion_labels[emotion_idx]} (Conf: {emotion_prob:.2f})",
            'Violence': f"{self.violence_labels[violence_idx]} (Conf: {violence_prob:.2f})",
            'Hate': f"{self.hate_labels[hate_idx]} (Conf: {hate_prob:.2f})"
        }
        
        return results

