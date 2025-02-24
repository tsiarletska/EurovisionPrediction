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

    




class EurovisionModel:
    def __init__(self, file_path, input_shape=None): #Added input_shape = None
        self.df = pd.read_csv(file_path)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.input_shape = input_shape #added this line.
    
    def preprocess_data(self):
        features = ["Year", "Country", "Performer", "Song", "Cleaned Lyrics", "Language", "Topic"]
        target = "Place"

        # Verify required columns exist
        missing_columns = [col for col in features + [target] if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert categorical features to strings and handle missing values
        categorical_features = ["Year", "Country", "Performer", "Song", "Language", "Topic"]
        self.df[categorical_features] = self.df[categorical_features].astype(str).fillna("Unknown")

        # One-Hot Encoding
        X_cat = self.encoder.fit_transform(self.df[categorical_features])

        # Lyrics Processing
        self.df['Cleaned Lyrics'] = self.df['Cleaned Lyrics'].astype(str).replace("", np.nan).dropna()
        X_lyrics = self.vectorizer.fit_transform(self.df["Cleaned Lyrics"]).toarray() * 2

        # Combine features
        X = np.hstack((X_cat, X_lyrics))
        y = self.df[target].values.flatten()

        # Scale numerical features
        X = self.scaler.fit_transform(X)

        # Train-test split
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def build_model(self, activation_function="sigmoid", learning_rate=0.1):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation=activation_function, input_shape=(self.input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation=activation_function),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation=activation_function),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation=activation_function),
            tf.keras.layers.Dense(1)
        ])

        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def train_and_evaluate(self, activation_function, learning_rate, batch_size, epochs):
        model = self.build_model(activation_function, learning_rate)
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predictions
        y_pred = model.predict(self.X_test).flatten()

        # Metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        spearman_corr, _ = spearmanr(self.y_test, y_pred)

        return {"activation": activation_function, "lr": learning_rate, "batch_size": batch_size,
                "epochs": epochs, "MAE": mae, "MSE": mse, "Spearman": spearman_corr}

    def run_hyperparameter_tuning(self, param_grid):
        results = []

        for activation in param_grid["activation"]:
            for lr in param_grid["learning_rate"]:
                for batch_size in param_grid["batch_size"]:
                    for epochs in param_grid["epochs"]:
                        result = self.train_and_evaluate(activation, lr, batch_size, epochs)
                        results.append(result)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("hyperparameter_results.csv", index=False)
        print("✅ Hyperparameter tuning complete. Results saved!")



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
        output_df.to_csv("results.csv", index=False)

        print("\n✅ Predictions saved to predictions_output.csv")
        
        return {
            "MAE": mae, 
            "MSE": mse, 
            "Spearman": spearman_corr,
            "Comparison": output_df
        }







# 1. Process the lyrics and create the processed dataset
processor = LyricsProcessor("data.csv")
processor.process()  # Creates processed_data.csv

# 2. Train the Eurovision Model
file_path = "processed_data.csv"
eurovision = EurovisionModel(file_path)

# 3. Preprocess training data
X_train, X_test, y_train, y_test = eurovision.preprocess_data()

# Add train and test data to the model instance
eurovision.X_train = X_train
eurovision.X_test = X_test
eurovision.y_train = y_train
eurovision.y_test = y_test

# Add this line to set the input shape
eurovision.input_shape = X_train.shape[1]

# 4. Define hyperparameter tuning grid
activation_functions = ["leaky_relu"]  #  "", "" ,  "leaky_relu"- add in the list later
#  "tanh" - done, elu,  sigmoid
learning_rates = [0.1,0.01, 0.001]   # , 0.01, 0.001
batch_sizes = [10,20,50,70,100]            #  [10, 20, 50, 70, 100]
epochs_list = [10,30,50,100,200]        #    [10, 30, 50, 100, 200]

# 5. Prepare results storage
results = []

# 6. Hyperparameter tuning loop
for activation, lr, batch_size, epochs in itertools.product(activation_functions, learning_rates, batch_sizes, epochs_list):
    #model_obj = EurovisionModel(file_path, input_shape=X_train.shape[1])  # Remove this line
    model = eurovision.build_model(activation_function=activation, learning_rate=lr)
    result = eurovision.train_and_evaluate(activation, lr, batch_size, epochs) #use eurovision instance
    results.append(result)

# 7. Save hyperparameter tuning results
results_df = pd.DataFrame(results)
results_df.to_csv("hyperparameter_results.csv", index=False)
print("✅ Hyperparameter tuning complete. Results saved!")

# 8. Train the final model using the best hyperparameters
best_params = results_df.nsmallest(1, "MAE").iloc[0]  # Choose best params based on MAE
best_model = eurovision.build_model(best_params["activation"], best_params["lr"])  # chaned here from "Learing Rate" to "learning_rate"
best_model.fit(eurovision.X_train, eurovision.y_train, epochs=int(best_params["epochs"]), batch_size=int(best_params["batch_size"]), verbose=0)

# 9. Make Predictions and Evaluate
test_file = "test_2024.csv"  # This file has all needed data including Place column
predictor = EurovisionTestPredictor(test_file, best_model, eurovision.vectorizer, eurovision.scaler, eurovision.encoder)

# 10. Make predictions
df_predictions = predictor.predict()

# 11. Evaluate model
results = predictor.evaluate()
