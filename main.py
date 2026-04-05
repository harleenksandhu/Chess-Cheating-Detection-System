from feature_extraction import extract_game_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_baseline_model(input_csv):
    df = pd.read_csv(input_csv) # load the processed dataset 

    X = df.drop(columns=['game_id', 'is_cheater', 'engine_move_rate', 'side']) 
    y = df['is_cheater']
    groups = df['game_id'] # used for grouped split

    # Train/Test split is 80-20
    grouped_split = GroupShuffleSplit(n_splits = 1, train_size=0.8, random_state=0)
    train_idx, test_idx = next(grouped_split.split(X, y, groups = groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


    # Random forest model training
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    random_forest.fit(X_train, y_train)

    # Predict
    y_pred = random_forest.predict(X_test)

    # Evaluation
    print("\n BASELINE MODEL PERFORMANCE")
    print(classification_report(y_test, y_pred))

    # Visualization of feature importance
    importances = random_forest.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.show()

def build_lstm_model(input_shape):
    model = Sequential([
        # Layer 1: Masking layer to tell model to ignore '0' padding that ensures games are same length
        Masking(mask_value=0.0, input_shape = input_shape), 

        # Layer 2: LSTM layer with 64 units
        LSTM(64, return_sequences=False), 

        # Layer 3: Dropout to prevent the model from overfitting
        Dropout(0.2), 

        # Layer 4: Decision layer (using sigmoid)
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return model

def create_lstm_sequences(input_csv, max_moves=50):
    df = pd.read_csv(input_csv)
    
    # cp_loss and move_time are the only features that change move-to-move
    # so the LSTM can watch over them
    features = ['cp_loss', 'move_time']
    
    X_list = []
    y_list = []
    elo_list = []

    # Group by game and player to treat every player as a unique sequence
    grouped = df.groupby(['game_id', 'player_elo'])
    
    for (_, elo), group in grouped: # (game_id, player_elo) in each group
        
        group = group.sort_values('move_index') # ensure chronological order
        
        # Extract features as a 2D array [moves, features]
        X_list.append(group[features].values)
        
        # The label is the same for the whole sequence (1 = cheater)
        y_list.append(group['is_cheater'].iloc[0])

        # Store the elo for this sequence
        elo_list.append(elo)
        
    # Post padding to ensure every game is exactly 50 moves long
    X = pad_sequences(X_list, maxlen=max_moves, dtype='float32', padding='post', truncating='post')
    
    return X, np.array(y_list), np.array(elo_list)


if __name__ == "__main__":
    # Extracting game features and processing the dataset
    # This creates a new csv file with the processed data
    '''
    input_csv = r"data\final_training_data.csv"
    output_csv = r"data\processed_data.csv"
    extract_game_features(input_csv, output_csv) # defined in feature_extraction.py
    
    '''
    # Baseline model
    #train_baseline_model(r"data\processed_data.csv")

    # LSTM model
    X, y, elos = create_lstm_sequences(r"data\final_training_data.csv")
    X_train, X_test, y_train, y_test, elos_train, elos_test = train_test_split(X, y, elos, test_size=0.2, random_state=0)
    lstm_model = build_lstm_model((50, 2))
    history = lstm_model.fit(X_train, y_train, epochs = 15, batch_size=64, validation_data = (X_test, y_test), verbose=1)

    # LSTM model performance
    y_pred = (lstm_model.predict(X_test) > 0.5).astype("uint32")
    print("\nLSTM MODEL PERFORMANCE")
    print(classification_report(y_test, y_pred))


    # Accuracy across different Elo ranges
    results_df = pd.DataFrame({'true': y_test, 'pred': y_pred.flatten(), 'elo': elos_test})

    def get_bin(e):
        if e < 1200: return 'Low (<1200)'
        if e < 1800: return 'Med (1200-1800)'
        return 'High (>1800)'
    
    results_df['bin'] = results_df['elo'].apply(get_bin)

    bin_analysis = results_df.groupby('bin').apply(lambda x: accuracy_score(x['true'], x['pred']))
    bin_analysis.plot(kind='bar', color='teal', title='Model Accuracy Across Elos')
    plt.ylabel('Accuracy')
    plt.show()

    # Accuracy and Loss Plots
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.show()

    lstm_model.save('chess_cheating_detector.keras')
    