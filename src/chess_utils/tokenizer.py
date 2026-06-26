import chess, numpy as np
from copy import copy
import torch.nn as nn

HUB          = 64       # Hub Idx
PIECE_DIM    = 12
CASTLING_DIM = 4
EP_DIM       = 16
COUNT_DIM    = 2        # full/half moves
GLOB_DIM     = CASTLING_DIM + EP_DIM + COUNT_DIM          # 22
F_DIM        = PIECE_DIM + GLOB_DIM                       # 34
N_NODES      = 65
EDGE_DIM     = 7

_piece_to_index = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                   chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}

# for numerical tokenizer
_piece_to_value = {chess.PAWN: 1/40, chess.KNIGHT: 3/40, chess.BISHOP: 3/40,
                   chess.ROOK: 5/40, chess.QUEEN: 9/40, chess.KING: 40/40}
_castling_val = 0.5/40
_enpassant_val = 0.1/40

# hub edges (64→hub and hub→64)
_HUB_SRC  = np.concatenate([np.full(64, HUB,  np.int16),
                            np.arange(64, dtype=np.int16)])
_HUB_DST  = np.concatenate([np.arange(64, dtype=np.int16),
                            np.full(64, HUB,  np.int16)])
_HUB_ATTR = np.zeros((128, EDGE_DIM), np.uint8)   #

# EDGE attributes (,7):     [legal,capture,defend,promo,opp,
#                           source_flag(1 for from src, 0 for from dis),
#                           (0 for gloabal, 1 for local)]

def _scaled_counts(brd: chess.Board):
    return brd.halfmove_clock / 100.0, brd.fullmove_number / 100.0


def build_tokenizer(config):
    if isinstance(config, dict):
        if config.get('numeric_input', False):
            return numerical_tokenize
        else:
            return tokenize
    elif isinstance(config, nn.Module):
        if hasattr(config, 'numeric_input') and config.numeric_input:
            return numerical_tokenize
        else:
            return tokenize
    else:
        raise ValueError(f'Unexpected type {type(config)}')

# ── tokenizer ────────────────────────────────────────────────────────────
def tokenize(fen: str):
    board = chess.Board(fen)
    me    = board.turn

    # node features -----------------------------------------------------
    x = np.zeros((N_NODES, F_DIM), np.uint8)

    for sq, pc in board.piece_map().items():
        idx = _piece_to_index[pc.piece_type] + (6 if pc.color != me else 0)
        x[sq, idx] = 1

    g = np.zeros(GLOB_DIM, np.float32)
    # castling rights (KQkq)
    g[0] = board.has_kingside_castling_rights(me)
    g[1] = board.has_queenside_castling_rights(me)
    g[2] = board.has_kingside_castling_rights(not me)
    g[3] = board.has_queenside_castling_rights(not me)
    # en-passant 16-bit file encoding
    if board.ep_square is not None:
        ep_idx = board.ep_square - (16 if board.ep_square < 32 else 40)
        g[4 + ep_idx] = 1
    # scaled clocks
    g[-2],g[-1]= _scaled_counts(board)
    x[HUB, PIECE_DIM:] = g


    #edge lists -----------------------------------------------
    src, dst, attr, seen = [], [], [], set()
    def add(u, v, f):
        if (u, v) not in seen:
            seen.add((u, v))
            src.append(u); dst.append(v); attr.append(f)

    LEGAL = np.array([1,0,0,0,0,1,1], np.uint8)
    BASE  = np.array([0,0,0,0,0,1,1], np.uint8)

    # legal moves for BOTH colours
    for colour in (me, not me):
        board.turn = colour
        opp_flag   = int(colour != me)
        for mv in board.legal_moves:
            f = LEGAL.copy()
            f[1] = board.is_capture(mv)
            f[3] = mv.promotion is not None
            f[4] = opp_flag
            add(mv.from_square, mv.to_square, f)
            f_rev = copy(f)
            f_rev[5] = 0
            add(mv.to_square,   mv.from_square, f_rev)

    # pseudo-attacks / defends
    for colour in (me, not me):
        board.turn = colour
        opp_flag   = int(colour != me)
        for s, pc in board.piece_map().items():
            for d in board.attacks(s):
                if (s, d) in seen: continue
                f = BASE.copy()
                f[1] = board.color_at(d) is not None and board.color_at(d) != colour  # capture?
                f[2] = board.color_at(d) == colour  # defend?
                f[4] = opp_flag
                add(s, d, f)
                f_rev = copy(f)
                f_rev[5] = 0
                add(d, s, f_rev)


    all_src = np.concatenate([np.asarray(src), _HUB_SRC])
    all_dst = np.concatenate([np.asarray(dst), _HUB_DST])
    edge_attr = np.concatenate([np.asarray(attr), _HUB_ATTR], axis=0).astype(np.float32)
    edge_index = np.vstack([all_src, all_dst])

    return x.astype(np.float32), edge_index, edge_attr




# ── numerical_tokenize ────────────────────────────────────────────────────────────
def numerical_tokenize(fen: str):
    board = chess.Board(fen)
    me    = board.turn

    # node features -----------------------------------------------------
    x = np.zeros((N_NODES, F_DIM), np.float32)

    for sq, pc in board.piece_map().items():
        idx = _piece_to_index[pc.piece_type] + (6 if pc.color != me else 0)
        x[sq, idx] = _piece_to_value[pc.piece_type]

    g = np.zeros(GLOB_DIM, np.float32)
    # castling rights (KQkq)
    g[0] = board.has_kingside_castling_rights(me) * _castling_val
    g[1] = board.has_queenside_castling_rights(me) * _castling_val
    g[2] = board.has_kingside_castling_rights(not me) * _castling_val
    g[3] = board.has_queenside_castling_rights(not me) * _castling_val
    # en-passant 16-bit file encoding
    if board.ep_square is not None:
        ep_idx = board.ep_square - (16 if board.ep_square < 32 else 40)
        g[4 + ep_idx] = _enpassant_val
    # scaled clocks
    g[-2],g[-1]= _scaled_counts(board)
    x[HUB, PIECE_DIM:] = g


    #edge lists -----------------------------------------------
    src, dst, attr, seen = [], [], [], set()
    def add(u, v, f):
        if (u, v) not in seen:
            seen.add((u, v))
            src.append(u); dst.append(v); attr.append(f)

    # LEGAL = np.array([1,0,0,0,0,1,1], np.float32)
    BASE  = np.array([0,0,0,0,0,1,1], np.float32)

    # legal moves for BOTH colours
    for colour in (me, not me):
        board.turn = colour
        opp_flag   = int(colour != me)
        for mv in board.pseudo_legal_moves:
            f = BASE.copy()
            if board.is_legal(mv):
                f[0] = 1.0
            if board.is_capture(mv):
                # f[1] = 1.0
                if board.is_en_passant(mv):
                    offset = -8 if board.turn == chess.WHITE else 8
                    captured_sq = mv.to_square + offset
                else:
                    captured_sq = mv.to_square
                f[1] = _piece_to_value[board.piece_at(captured_sq).piece_type]
            f[3] = mv.promotion is not None
            f[4] = opp_flag
            add(mv.from_square, mv.to_square, f)
            f_rev = copy(f)
            f_rev[5] = 0
            add(mv.to_square,   mv.from_square, f_rev)

    # pseudo-attacks / defends
    for colour in (me, not me):
        board.turn = colour
        opp_flag   = int(colour != me)
        for s, pc in board.piece_map().items():
            for d in board.attacks(s):
                if (s, d) in seen: continue
                f = BASE.copy()
                f[2] = board.color_at(d) == colour  # defend?
                f[4] = opp_flag
                add(s, d, f)
                f_rev = copy(f)
                f_rev[5] = 0
                add(d, s, f_rev)


    all_src = np.concatenate([np.asarray(src), _HUB_SRC])
    all_dst = np.concatenate([np.asarray(dst), _HUB_DST])
    edge_attr = np.concatenate([np.asarray(attr), _HUB_ATTR], axis=0).astype(np.float32)
    edge_index = np.vstack([all_src, all_dst])

    return x.astype(np.float32), edge_index, edge_attr