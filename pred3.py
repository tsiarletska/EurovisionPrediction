import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from langid_fasttext import LanguageIdentifier  # FastText-based language detector


def clean_lyrics(lyrics):
    # Remove text in square brackets and convert to lowercase
    cleaned = re.sub(r'\[.*?\]', '', str(lyrics))
    return cleaned.lower()



# Load datasets
data = pd.read_csv("data.csv")
test_data = pd.read_csv("test_2024.csv")

# Clean lyrics
data["Lyrics"] = data["Lyrics"].apply(clean_lyrics)
test_data["Lyrics"] = test_data["Lyrics"].apply(clean_lyrics)


# Convert lyrics to numerical representation (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_lyrics = vectorizer.fit_transform(data["Lyrics"]).toarray()
X_test_lyrics = vectorizer.transform(test_data["Lyrics"]).toarray()

# Our features are now just the lyrics vectors
X = X_lyrics
y = data["Place"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test_lyrics)
y_test = test_data["Place"].values

# Split training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network model
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
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

# Add early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Predict on test set
y_pred = model.predict(X_test).flatten()

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest Set Metrics:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
print(f"R2 Score: {r2:.3f}")

# Create detailed results DataFrame
results_df = pd.DataFrame({
    'Country': test_data['Country'],
    'Song': test_data['Song'],
    'Performer': test_data['Performer'],
    'Actual_Place': y_test,
    'Predicted_Place': np.round(y_pred, 1),
    'Prediction_Error': np.abs(y_test - y_pred)
})

# Sort by prediction error to see where model was most/least accurate
results_df = results_df.sort_values('Prediction_Error')

# Display results
print("\nDetailed Predictions:")
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('eurovision_predictions.csv', index=False)

# Optional: Plot predictions vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Place')
plt.ylabel('Predicted Place')
plt.title('Predicted vs Actual Eurovision Places')
plt.tight_layout()
plt.show()