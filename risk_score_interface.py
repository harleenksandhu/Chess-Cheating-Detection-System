import matplotlib.pyplot as plt
import numpy as np

def plot_game_risk_profile(model, game_sequence, elo, side):
    ''''
    Takes a single game sequence and plots the risk move-by-move.
    '''
    risks = []
    
    # Calculate risk at each move 
    for i in range(1, len(game_sequence) + 1):
        # Creating a partial sequence (padding the rest with zeros)
        current_seq = np.zeros_like(game_sequence)
        current_seq[:i] = game_sequence[:i]
        
        # Get probability from model
        prob = model.predict(current_seq.reshape(1, len(game_sequence), 2), verbose=0)[0][0]
        risks.append(prob)
    
    # Plotting the Risk Profile
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(game_sequence) + 1), risks, marker='o', linestyle='-', linewidth=2, color='red') # Risk plot
    plt.axhline(y=0.5, color='black', linestyle='--', label='Suspicion Threshold') # Suspcion Threshold line
    plt.fill_between(range(1, len(game_sequence) + 1), risks, 0.5, where=(np.array(risks) >= 0.5), label='Cheat Zone', color='red', alpha=0.3) # Highlights area under line and above threshold
    plt.title(f"Cheating Risk Profile: {side} Player ({elo} Elo)")
    plt.xlabel("Move Number")
    plt.ylabel("Risk Score (Probability)")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
