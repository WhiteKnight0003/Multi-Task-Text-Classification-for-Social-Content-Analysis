from tensorflow import keras
import numpy as np

class MultiTaskModel:
    """Multi-task learning model"""
    
    def __init__(self, vocab_size, max_length, num_emotion_classes, num_violence_classes, num_hate_classes):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_emotion_classes = num_emotion_classes
        self.num_violence_classes = num_violence_classes
        self.num_hate_classes = num_hate_classes
        self.model = None
    
    def build_model(self):
        # Input layers
        emotion_input = keras.layers.Input(shape=(self.max_length,), name='emotion_input')
        violence_input = keras.layers.Input(shape=(self.max_length,), name='violence_input')
        hate_input = keras.layers.Input(shape=(self.max_length,), name='hate_input')
        
        # Shared embedding layer
        embedding_layer = keras.layers.Embedding(
            input_dim=self.vocab_size, 
            output_dim=128
        )
        
        # Apply embedding
        emotion_embedding = embedding_layer(emotion_input)
        violence_embedding = embedding_layer(violence_input)
        hate_embedding = embedding_layer(hate_input)
        
        # Shared LSTM layer
        shared_lstm = keras.layers.LSTM(64, return_sequences=True)
        
        emotion_lstm = shared_lstm(emotion_embedding)
        violence_lstm = shared_lstm(violence_embedding)
        hate_lstm = shared_lstm(hate_embedding)
        
        # Shared pooling and dropout
        shared_pooling = keras.layers.GlobalAveragePooling1D()
        shared_dropout = keras.layers.Dropout(0.5)
        
        emotion_features = shared_dropout(shared_pooling(emotion_lstm))
        violence_features = shared_dropout(shared_pooling(violence_lstm))
        hate_features = shared_dropout(shared_pooling(hate_lstm))
        
        # Output layers
        emotion_output = keras.layers.Dense(
            self.num_emotion_classes, 
            activation='softmax', 
            name='emotion_output'
        )(emotion_features)
        
        violence_output = keras.layers.Dense(
            self.num_violence_classes,
            activation='softmax',
            name='violence_output'
        )(violence_features)
        
        hate_output = keras.layers.Dense(
            self.num_hate_classes,
            activation='softmax',
            name='hate_output'
        )(hate_features)
        
        # Create model
        self.model = keras.models.Model(
            inputs=[emotion_input, violence_input, hate_input],
            outputs=[emotion_output, violence_output, hate_output]
        )
        
        return self.model
    
    def compile_model(self):
        """Compile the model"""
        self.model.compile(
            optimizer='adam',
            loss={
                'emotion_output': 'sparse_categorical_crossentropy',
                'violence_output': 'sparse_categorical_crossentropy',
                'hate_output': 'sparse_categorical_crossentropy'
            },
            metrics={
                'emotion_output': 'accuracy',
                'violence_output': 'accuracy',
                'hate_output': 'accuracy'
            }
        )
    
    def train(self, processed_data, epochs=10, batch_size=16, model_save_path='best_model.weights.h5'):
        """Train the model"""

        # 1. ModelCheckpoint:
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,     
            monitor='val_loss',          
            save_best_only=True,      
            save_weights_only=True,     
            mode='min',                 
            verbose=1                 
        )
        
        # 2. EarlyStopping:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,                 
            restore_best_weights=True   # <--- TỰ ĐỘNG KHÔI PHỤC WEIGHTS TỐT NHẤT
        )

        validation_x = {
            'emotion_input': processed_data['emotion']['x_test'],
            'violence_input': processed_data['violence']['x_test'],
            'hate_input': processed_data['hate']['x_test']
        }
        validation_y = {
            'emotion_output': processed_data['emotion']['y_test'],
            'violence_output': processed_data['violence']['y_test'],
            'hate_output': processed_data['hate']['y_test']
        }

        history = self.model.fit(
            x={
                'emotion_input': processed_data['emotion']['x_train'],
                'violence_input': processed_data['violence']['x_train'],
                'hate_input': processed_data['hate']['x_train']
            },
            y={
                'emotion_output': processed_data['emotion']['y_train'],
                'violence_output': processed_data['violence']['y_train'],
                'hate_output': processed_data['hate']['y_train']
            },
            validation_data=(validation_x, validation_y), 
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping]
        )
        return history
    
    def predict(self, processed_data):
        """Make predictions"""
        predictions = self.model.predict({
            'emotion_input': processed_data['emotion']['x_test'],
            'violence_input': processed_data['violence']['x_test'],
            'hate_input': processed_data['hate']['x_test']
        })
        
        emotion_pred = np.argmax(predictions[0], axis=1)
        violence_pred = np.argmax(predictions[1], axis=1)
        hate_pred = np.argmax(predictions[2], axis=1)
        
        return emotion_pred, violence_pred, hate_pred


    def load_weights(self, model_save_path):
        if not self.model:
            print("Model phải được build và compile trước khi load weights.")
            return
        
        print(f"Đang tải pre-trained weights từ {model_save_path}...")
        self.model.load_weights(model_save_path)
        print("Tải weights thành công.")