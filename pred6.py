import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

def clean_lyrics(lyrics):
    # Remove text in square brackets and convert to lowercase
    cleaned = re.sub(r'\[.*?\]', '', str(lyrics))
    return cleaned.lower()

# Load datasets
data = pd.read_csv("data.csv")
new_data = pd.read_csv("test_2025.csv")  # New dataset without actual placements

# Clean lyrics
data["Lyrics"] = data["Lyrics"].apply(clean_lyrics)
new_data["Lyrics"] = new_data["Lyrics"].apply(clean_lyrics)

# Convert lyrics to numerical representation (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_lyrics = vectorizer.fit_transform(data["Lyrics"]).toarray()
X_new_lyrics = vectorizer.transform(new_data["Lyrics"]).toarray()

# Our features are now just the lyrics vectors
X = X_lyrics
y = data["Place"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_new = scaler.transform(X_new_lyrics)

# Build neural network model
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Regression output
])

# Compile model with learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# Predict placements for new data
y_pred_new = model.predict(X_new).flatten()

# Create results DataFrame
results_df = pd.DataFrame({
    'Country': new_data['Country'],
    'Song': new_data['Song'],
    'Performer': new_data['Performer'],
    'Predicted_Place': np.round(y_pred_new, 1)
})

# Save results to CSV and ensure it's written
results_csv_path = 'predicted_2025.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")

# Display results
print("\nPredicted Eurovision Placements:")
print(results_df.to_string(index=False))
