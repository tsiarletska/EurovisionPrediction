import pandas as pd
import numpy as np
import re
from langdetect import detect_langs
from bertopic import BERTopic
import tensorflow as tf


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from scipy.stats import spearmanr


import itertools





# cleansing of the lyrics, detection of the languages (up to 3 languages), and estimation of the topics using BERTopic.
# saves data to a new CSV file called processed_data.csv.
# columns to pass to the X: Place, Performer, Song, Cleaned_Lyrics, Language, Topic
class LyricsProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.topic_model = None  # Initialize but don't fit yet
    
    # def clean_lyrics(self, lyrics):
    #     """Removes text inside square brackets and converts to lowercase."""
    #     cleaned = re.sub(r'\[.*?\]', '', str(lyrics))
    #     return cleaned.lower()



    def clean_lyrics(self, lyrics):
        try:
            cleaned = re.sub(r'\[.*?\]', '', str(lyrics))  # Ensure string conversion happens *inside* the try
            cleaned = cleaned.lower()
            return cleaned
        except: # Handle non-string input
            return ""  # Return empty string if cleaning fails

    
    def detect_languages(self, lyrics):
        """Detects up to 3 languages in the lyrics."""
        try:
            detected = detect_langs(lyrics)
            languages = [lang.lang for lang in detected[:3]]  # Get top 3 languages
            return ', '.join(languages)
        except:
            return "Unknown"
    
    def fit_topic_model(self):
        """Fits BERTopic on the entire dataset."""
        cleaned_lyrics = self.df['Cleaned Lyrics'].dropna().tolist()
        if len(cleaned_lyrics) > 1:
            self.topic_model = BERTopic()
            self.topics, _ = self.topic_model.fit_transform(cleaned_lyrics)
        else:
            self.topic_model = None  # Not enough data to train
    
    def estimate_topic(self, lyrics):
        """Estimates the topic of the lyrics using BERTopic."""
        if self.topic_model:
            topic, _ = self.topic_model.transform([lyrics])
            return topic[0]
        return "Unknown"
    
    def process(self):
        """Cleans lyrics, detects languages, and estimates topics."""
        self.df['Cleaned Lyrics'] = self.df['Lyrics'].apply(self.clean_lyrics)
        self.df['Language'] = self.df['Cleaned Lyrics'].apply(self.detect_languages)
        
        # Fit topic model before topic estimation
        self.fit_topic_model()
        
        self.df['Topic'] = self.df['Cleaned Lyrics'].apply(self.estimate_topic)
        
        output_file = "processed_data.csv"
        self.df.to_csv(output_file, index=False)
        print(f"Processed file saved as {output_file}")

    



# previos 
class EurovisionModel:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def preprocess_data(self):
        # Select relevant columns
        features = ["Year", "Country", "Performer", "Song", "Cleaned Lyrics", "Language", "Topic"]
        target = "Place"
        
        # Verify required columns exist
        missing_columns = [col for col in features + [target] if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle categorical data
        categorical_features = ["Year", "Country", "Performer", "Song", "Language", "Topic"]

        # 1. Convert ALL categorical columns to strings *FIRST*
        for col in categorical_features:
            self.df[col] = self.df[col].astype(str)

        # 2. Handle missing values *before* one-hot encoding
        for col in categorical_features:
            self.df[col] = self.df[col].fillna("Unknown")  # Or another appropriate placeholder

        # 3. One-Hot Encode
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Re-initialize encoder
        X_cat = self.encoder.fit_transform(self.df[categorical_features]) # Fit on train data
        
        # 4. Lyrics Handling (Force string, remove empty or NaN)
        self.df['Cleaned Lyrics'] = self.df['Cleaned Lyrics'].astype(str)
        self.df = self.df[self.df['Cleaned Lyrics'] != ""]  # Remove empty strings
        self.df = self.df.dropna(subset=['Cleaned Lyrics'])  # Remove NaN values
        X_lyrics = self.vectorizer.fit_transform(self.df["Cleaned Lyrics"]) # Fit on train data
        X_lyrics = X_lyrics.toarray() * 2
        
        # Combine features
        X = np.hstack((X_cat, X_lyrics))
        y = self.df[target].values.flatten()
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='sigmoid', input_shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation='sigmoid'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ])
        
        # Learning rate schedule
        initial_learning_rate = 0.1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile model
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test):
        model = self.build_model(X_train.shape[1])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=10,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return model




# # === Usage ===
# # Load Data and Preprocess
# file_path = "processed_data.csv"
# eurovision = EurovisionModel(file_path)
# X_train, X_test, y_train, y_test = eurovision.preprocess_data()

# # Hyperparameter Grid
# param_grid = {
#     "activation": ["relu", "sigmoid"],
#     "learning_rate": [0.1, 0.01],
#     "batch_size": [10, 20],
#     "epochs": [10, 20]
# }

# # Run Hyperparameter Tuning
# eurovision.run_hyperparameter_tuning(param_grid)


class EurovisionTestPredictor:
    def __init__(self, test_file, trained_model, vectorizer, scaler, encoder):
        self.df = pd.read_csv(test_file)
        self.model = trained_model
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.encoder = encoder
    
    def clean_lyrics(self, lyrics):
        """Removes text inside square brackets and converts to lowercase."""
        return re.sub(r'\[.*?\]', '', str(lyrics)).lower()
    
    def detect_languages(self, lyrics):
        """Detects up to 3 languages in the lyrics."""
        try:
            detected = detect_langs(lyrics)
            return ', '.join([lang.lang for lang in detected[:3]])
        except:
            return "Unknown"
    
    def preprocess_data(self):
            """Preprocess test data similarly to training data."""
            
            self.df['Cleaned Lyrics'] = self.df['Lyrics'].apply(self.clean_lyrics)
            self.df['Language'] = self.df['Cleaned Lyrics'].apply(self.detect_languages)

            # 👉 Store "Place" separately and remove from input features
            self.df["Actual_Place"] = self.df["Place"] if "Place" in self.df.columns else None
            self.df.drop(columns=["Place"], inplace=True, errors="ignore")

            categorical_features = ["Year", "Country", "Performer", "Song", "Language", "Topic"]

            # Convert categorical columns to strings and handle missing values
            for col in categorical_features:
                self.df[col] = self.df[col].astype(str).fillna("Unknown")

            # One-Hot Encode categorical data using the pre-trained encoder
            X_cat = self.encoder.transform(self.df[categorical_features])

            # Lyrics processing
            self.df['Cleaned Lyrics'] = self.df['Cleaned Lyrics'].astype(str)
            self.df = self.df[self.df['Cleaned Lyrics'] != ""]
            self.df = self.df.dropna(subset=['Cleaned Lyrics'])

            X_lyrics = self.vectorizer.transform(self.df["Cleaned Lyrics"])
            X_lyrics = X_lyrics.toarray() * 2  # Keep the transformation consistent with training

            # Combine features
            X = np.hstack((X_cat, X_lyrics))
            X = self.scaler.transform(X)
            
            return X
    
    def predict(self):
        """Predict the placement of the songs."""
        X_test = self.preprocess_data()
        predictions = self.model.predict(X_test).flatten()
        
        self.df['Predicted_Place'] = predictions

        return self.df
    
    def evaluate(self):
        """Evaluates predictions against actual Places."""
        if self.df["Actual_Place"] is None:
            print("⚠️ No actual placements available in test data.")
            return
        
        y_true = self.df['Actual_Place'].astype(float)
        y_pred = self.df['Predicted_Place']

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)

        print("\n📊 Evaluation Results:")
        print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
        print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
        print(f"🔹 Spearman Rank Correlation: {spearman_corr:.4f}")

        # Save results to CSV
        output_df = self.df[['Song', 'Performer', 'Year', 'Predicted_Place', 'Actual_Place']]
        output_df.to_csv("results_pred4.csv", index=False)

        print("\n✅ Predictions saved to predictions_output.csv")
        
        return {
            "MAE": mae, 
            "MSE": mse, 
            "Spearman": spearman_corr,
            "Comparison": output_df
        }

# Usage:
# predictor = EurovisionTestPredictor("test_2024.csv", model, vectorizer, scaler, encoder)
# predictor.predict()
# predictor.evaluate("test_2024_actual.csv")



# 1. Process the lyrics and create the processed dataset
processor = LyricsProcessor("data.csv")
processor.process()  # Creates processed_data.csv

# 2. Train the Eurovision Model
file_path = "processed_data.csv"
eurovision = EurovisionModel(file_path)

# Preprocess training data
X_train, X_test, y_train, y_test = eurovision.preprocess_data()

# Train the model
model = eurovision.train_model(X_train, X_test, y_train, y_test)

# 3. Make Predictions and Evaluate
test_file = "test_2024.csv"  # This file has all needed data including Place column
predictor = EurovisionTestPredictor(test_file, model, eurovision.vectorizer, eurovision.scaler, eurovision.encoder)

# Make predictions and evaluate using the Place column from the same file
df_predictions = predictor.predict()
results = predictor.evaluate()  # Now this works without parameters

# Print results
# print("\nPredictions and Evaluation Results:")
# print(results)


