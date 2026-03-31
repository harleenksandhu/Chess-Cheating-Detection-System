import pandas as pd

def extract_game_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # creating separate DFs for white moves and black moves (since both can cheat)
    white_moves = df[df['move_index'] % 2 == 0].copy()
    black_moves = df[df['move_index'] % 2 != 0].copy()

    def get_features(data, colour):
        # the dataset was generated with each row being one move
        # so now each game's moves will be grouped together based on game_id'
        # to aggregate move-level data into game-level features
        features = data.groupby('game_id', as_index=False).agg({
            'is_cheater': 'max', # if any move was a cheat move, the player is a cheater 
            'player_elo': 'first', 
            'opponent_elo': 'first', 
            'elo_diff': 'first', # skill gap between black and white
            'cp_loss': ['mean', 'std'], # average centipawn loss and consistency
            'move_time': ['mean', 'std'], # average speed and rhythm
            'label': 'mean' # rate of engine moves 
        })
        # renaming columns
        features.columns = ['game_id', 'is_cheater', 'elo', 'opponent_elo', 'elo_diff', 'acpl', 'cpl_std', 'avg_time', 'time_std', 'engine_move_rate']
        features['side'] = colour
        return features

    white_features = get_features(white_moves, 'white')
    black_features = get_features(black_moves, 'black')

    final_dataset = pd.concat([white_features, black_features], ignore_index=True)
    final_dataset = final_dataset.sort_values(by=['game_id', 'side'], ascending=[True, False])
    final_dataset.to_csv(output_csv, index = False)

