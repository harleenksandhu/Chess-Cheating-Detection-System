from keras.models import load_model
from model_training import create_lstm_sequences
from risk_score_interface import plot_game_risk_profile

model = load_model("chess_cheating_detector.keras") # Loading model created in model_training.py
X, y, elos = create_lstm_sequences("data/test_data.csv", max_moves=30) # Only taking 30 moves per game

# Evaluating the risk scores from the test data (9 games)
for i in range(9):
    # Player 1
    plot_game_risk_profile(model, X[2*i], elos[2*i], "Black")

    # Player 2
    plot_game_risk_profile(model, X[2*i+1], elos[2*i+1], "White")

