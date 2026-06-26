import os

import torch
from torch_geometric.data import Batch

from src.chess_utils.data_loader import transform_to_data
from src.chess_utils.tokenizer import build_tokenizer
from src.basics import get_device
import torch.nn.functional as F



class ActionChooser():
    def __init__(self, path,model = None):
        self.device = get_device()
        if model is not None:
            self.model = model
        else:
            checkpoint = torch.load(path, weights_only=False,map_location=self.device)
            self.model = checkpoint["model"].to(self.device)
            self.config = checkpoint["config"]
            self.model_name = self.model.__class__.__name__

        self.tokenize_fn = build_tokenizer(self.model)


    @torch.no_grad()
    def play(self, board):
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]
        graphs = []
        for mv in legal_moves:
            board.push(mv)
            x,ei,ea = self.tokenize_fn(board.fen())
            graphs.append(transform_to_data(x,ei,ea))
            board.pop()

        batch = Batch.from_data_list(graphs).to(self.device)

        output = self.model(batch)
        probs = F.softmax(output, dim=1)
        choices = torch.argmin(probs, dim=0).tolist()
        # This is changed to min because It's the opponent's winning prob
        return legal_moves[choices[0]]




class EnsembleActionChooser():
    def __init__(self, path,ensemble):
        self.device = get_device()
        self.models = []
        for i in range(1, ensemble + 1):
            assert not hasattr(m, 'numeric_input')
            checkpoint = torch.load(os.path.join(path,f'model{i}.pth'), weights_only=False,map_location=self.device)
            m = checkpoint["model"].to(self.device)
            m.eval()
            self.models.append(m)
        self.tokenize_fn = build_tokenizer(m)

    @torch.no_grad()
    def play(self, board):
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]

        graphs = []
        for mv in legal_moves:
            board.push(mv)
            x, ei, ea = self.tokenize_fn(board.fen())
            graphs.append(transform_to_data(x, ei, ea))
            board.pop()
        batch = Batch.from_data_list(graphs).to(self.device)

        all_probs = []
        for m in self.models:
            out = m(batch)  # logits
            probs = F.softmax(out, dim=1)
            all_probs.append(probs)
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        choices = torch.argmin(avg_probs, dim=0).tolist()
        return legal_moves[choices[0]]


