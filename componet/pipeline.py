from componet.load_data import DataLoader
from componet.preprocessor import DataPreprocessor
from componet.split import DataSplitter
from componet.tokenizer import TextTokenizer
from componet.model import MultiTaskModel
from componet.evalution import ModelEvaluator, TextClassifier

class MultiTaskPipeline:
    """Complete pipeline for multi-task learning"""
    
    def __init__(self, emotion_path, violence_path, hate_path):
        self.emotion_path = emotion_path
        self.violence_path = violence_path
        self.hate_path = hate_path
        
        self.data_loader = None
        self.preprocessor = None
        self.splitter = None
        self.tokenizer = None
        self.model = None
        self.evaluator = None
        self.classifier = None
    
    def run(self, epochs=10, batch_size=16, model_save_path='best_model.weights.h5', mode='train'):
        """Run complete pipeline"""
        print("=" * 50)
        print("Starting Multi-Task Learning Pipeline")
        print("=" * 50)
        
        # Step 1: Load data
        self.data_loader = DataLoader(
            self.emotion_path, 
            self.violence_path, 
            self.hate_path
        )
        emotion_df, violence_df, hate_df = self.data_loader.load_data()

        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        self.preprocessor = DataPreprocessor()
        emotion_df, violence_df, hate_df = self.preprocessor.process(
            emotion_df, violence_df,  hate_df
        )
        
        # Step 3: Split data
        print("\n3. Splitting data...")
        self.splitter = DataSplitter()
        train_test_splits = self.splitter.split(emotion_df, violence_df, hate_df)
        
        # Step 4: Tokenize and pad
        print("\n4. Tokenizing and padding...")
        self.tokenizer = TextTokenizer()
        self.tokenizer.fit_tokenizer(
            train_test_splits['emotion'][0],
            train_test_splits['violence'][0],
            train_test_splits['hate'][0]
        )
        processed_data = self.tokenizer.process_splits(train_test_splits)
        
        # Step 5: Build and compile model
        print("\n5. Building model...")
        self.model = MultiTaskModel(
            vocab_size=len(self.tokenizer.tokenizer.word_index) + 1,
            max_length=self.tokenizer.max_length,
            num_emotion_classes=len(emotion_df['label'].unique()),
            num_violence_classes=len(violence_df['label'].unique()),
            num_hate_classes=len(hate_df['label'].unique())
        )
        self.model.build_model()
        self.model.compile_model()
        print(self.model.model.summary())
        
        # Step 6: Train model
        # print("\n6. Training model...")
        # self.model.train(processed_data, epochs=epochs, batch_size=batch_size)
        if mode == 'train':
            print("\n6. Huấn luyện model...")
            self.model.train(processed_data, epochs=epochs, batch_size=batch_size, model_save_path=model_save_path)
        
        elif mode == 'load':
            print("\n6. Tải model đã huấn luyện...")
            try:
                self.model.load_weights(model_save_path)
            except FileNotFoundError:
                print(f"LỖI: Không tìm thấy tệp model tại {model_save_path}.")
                return None
            except Exception as e:
                print(f"Lỗi khi tải weights: {e}")
                return None
        else:
            print(f"Lỗi: Chế độ '{mode}' không hợp lệ. Chỉ dùng 'train' hoặc 'load'.")
            return None
        
        # Step 7: Evaluate model
        print("\n7. Evaluating model...")
        emotion_pred, violence_pred, hate_pred = self.model.predict(processed_data)
        
        self.evaluator = ModelEvaluator()
        self.evaluator.evaluate(processed_data, emotion_pred, violence_pred, hate_pred)
        
        # Step 8: Create classifier
        print("\n8. Creating text classifier...")
        self.classifier = TextClassifier(
            self.model.model, 
            self.tokenizer, 
            self.preprocessor
        )
        
        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        print("=" * 50)
        
        return self.classifier
