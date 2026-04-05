import matplotlib.pyplot as plt
import numpy as np

# CLANKA CODE !!!!!!!!!!!!!!!!!

def plot_game_risk_profile(model, game_sequence, elo, side):
    """
    Takes a single game sequence [30, 2] and plots the risk move-by-move.
    """
    risks = []
    
    # Calculate risk at each move (from move 1 to 30)
    for i in range(1, len(game_sequence) + 1):
        # Create a partial sequence (pad the rest with zeros)
        current_seq = np.zeros_like(game_sequence)
        current_seq[:i] = game_sequence[:i]
        
        # Get probability from model
        prob = model.predict(current_seq.reshape(1, len(game_sequence), 2), verbose=0)[0][0]
        risks.append(prob)
    
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(game_sequence) + 1), risks, marker='o', linestyle='-', color='red', linewidth=2)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Suspicion Threshold')
    
    plt.fill_between(range(1, len(game_sequence) + 1), risks, 0.5, where=(np.array(risks) >= 0.5), 
                     color='red', alpha=0.3, label='Cheat Zone')
    
    plt.title(f"Cheating Risk Profile: {side} Player ({elo} Elo)")
    plt.xlabel("Move Number")
    plt.ylabel("Risk Score (Probability)")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Usage: Pick a known cheater game from your test set
# plot_game_risk_profile(model, X_test[10], "G_742")