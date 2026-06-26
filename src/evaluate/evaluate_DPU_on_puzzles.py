from collections.abc import Sequence
import io
import os
import chess
import chess.engine
import chess.pgn
import numpy as np

from tqdm import tqdm
import pandas as pd

# from src.evaluate.action_chooser_minmax import MinimaxActionChooser
from src.evaluate.plotting import plot_puzzle_results
from src.evaluate.action_chooser import ActionChooser, EnsembleActionChooser


def evaluate_puzzle_from_pandas_row(
    puzzle,
    engine
):
  """Returns True if the `engine` solves the puzzle and False otherwise."""
  game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
  if game is None:
    raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
  board = game.end().board()
  moves = puzzle['Moves'].split(' ')
  return (len(moves),
          evaluate_puzzle_from_board(
            board=board,
            moves = moves,
            engine=engine,
          ))


def evaluate_puzzle_from_board(
    board: chess.Board,
    moves: Sequence[str],
    engine
) -> bool:
  """Returns True if the `engine` solves the puzzle and False otherwise."""
  for move_idx, move in enumerate(moves):
    if move_idx % 2 == 1:
      predicted_move = engine.play(board).uci()
      if move != predicted_move:
        board.push(chess.Move.from_uci(predicted_move))
        return board.is_checkmate()
    board.push(chess.Move.from_uci(move))
  return True

def get_interval_str(value, start=200, end=3000, step=200):
  if value < start or value >= end:
    raise ValueError(f"Input {value} is out of range.")
  bucket = (value - start) // step
  lower = start + bucket * step
  return f"{lower}-{lower + step}"


def eval_DPU_on_puzzles(out_path,score_filter = None,ensemble = None,
                        model = None,minmax = False,minmax_depth = 2,minmax_top_k = None,
                        model_choice = None,save_name_suffix = None, chess_data_root = None):

  result_df = pd.DataFrame(columns=[f"{i}-{i + 200}" for i in range(200, 3000, 200)])
  result_df.index.name = 'puzzle_len'

  chess_root = chess_data_root or os.path.join(os.getcwd(), 'data/chess_data')
  puzzles_path = os.path.join(chess_root, 'puzzles.csv')
  puzzles = pd.read_csv(puzzles_path)
  if score_filter is not None:
    puzzles = puzzles[puzzles['Rating'] < score_filter]
    puzzle_result_name = f"puzzle_result_{str(score_filter)}"
  else:
    puzzle_result_name = f"puzzle_result"
  if save_name_suffix is not None:
    puzzle_result_name = f"{puzzle_result_name}_{save_name_suffix}"
    print(f"Saving results to {puzzle_result_name}")

  if ensemble is not None:
    raise NotImplementedError('Ensemble engine is Deprecated.')
    # engine = EnsembleActionChooser(out_path,ensemble)
  elif minmax:
    raise NotImplementedError('MinMax engine is Deprecated.')
    # if model is not None:
    #   engine = MinimaxActionChooser(out_path,model,depth = minmax_depth,top_k = minmax_top_k)
    # else:
    #   engine = MinimaxActionChooser(os.path.join(out_path, 'model.pth'),depth = minmax_depth,top_k = minmax_top_k)
    
  else:
    if model is not None:
      engine = ActionChooser(out_path,model)
    else:
      if model_choice == 'best_test_model':
        engine = ActionChooser(os.path.join(out_path, 'best_test_model.pth'))
      else:
        engine = ActionChooser(os.path.join(out_path, 'model.pth'))

  for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=len(puzzles), desc="Evaluating puzzles"):
    puzzle_len, correct = evaluate_puzzle_from_pandas_row(
      puzzle=puzzle,
      engine=engine,
    )
    interval = get_interval_str(puzzle['Rating'])
    if puzzle_len not in result_df.index:
      result_df.loc[puzzle_len] = {col: np.array([0,0]) for col in result_df.columns}
    result_df.loc[puzzle_len, interval] = result_df.loc[puzzle_len, interval] + [correct,1]
  
  if minmax:
    out_path = os.path.join(out_path, f'minmax_depth_{minmax_depth}' + ('_top_k_' + str(minmax_top_k) if minmax_top_k is not None else ''))
    os.makedirs(out_path, exist_ok=True)
  result_df.to_pickle(os.path.join(out_path, puzzle_result_name + '.pkl'))

  # plot and save barplot
  plot_puzzle_results(result_df, out_path, puzzle_result_name)

  return result_df


