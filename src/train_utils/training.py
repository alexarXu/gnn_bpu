import os
import pickle
import numpy as np
import torch
from torch import optim

import math
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, LinearLR, CosineAnnealingLR
from src import constants
from src.evaluate.evaluate_DPU_on_puzzles import eval_DPU_on_puzzles
from src.evaluate.plotting import plot_bar
from src.model.init_model import initialize_model
from src.train_utils.losses import soft_cross_entropy
from src.train_utils.epoch_train_et_eval import *
from src.basics import count_params, get_device, get_out_path
from src import chess_config as config_lib


def train(
    train_config: config_lib.TrainConfig,
    test_config: config_lib.EvalConfig,
    build_data_loader: constants.DataLoaderBuilder,
    exp_config: dict,
    ensemble = False
):
    """Trains a predictor and returns the trained parameters."""
    # Prefer the device from exp_config; otherwise auto-detect.
    device = exp_config.get('device', None)
    if device is None:
        device = get_device()
    else:
        # Fall back if the requested device is unavailable.
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU.")
            device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            print("MPS is unavailable; falling back to CPU.")
            device = 'cpu'
    
    print(f"Training device: {device}")
    continue_train = exp_config.get('continue_train', False)

    if continue_train:
        model_path = exp_config.get('checkpoint_path', None)
        if model_path is None:
            model_path = os.path.join(get_out_path(exp_config), f'model.pth')
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        exp_config['exp_id'] = exp_config['exp_id'] + '_from_checkpoint'
        print(f"Continue training from checkpoint {model_path}")

    else:
        print("Initializing model...")
        try:
            model = initialize_model(exp_config)
            print("Model initialization complete.")
        except Exception as e:
            print(f"Model initialization failed: {e}")
            raise e

    print("Counting model parameters...")
    params_ct = count_params(model)
    print("Model architecture:")
    print(model)
    print(f"Trainable parameters: {params_ct:,}")
    
    print(f"Moving model to device: {device}")
    model.to(device)
    print("Model moved to device.")

    seed = exp_config['seed'] + ensemble
    train_config.data.seed = seed
    test_config.data.seed = seed

    if continue_train:
        seed = seed + 12345
        print(f"Training seed {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    out_path = get_out_path(exp_config)
    print(f"Output path: {out_path}")
    os.makedirs(out_path, exist_ok=True)

    print("Creating training data loader...")
    try:
        train_loader = build_data_loader(config=train_config.data,exp_config = exp_config)
        print("Training data loader ready.")
    except Exception as e:
        print(f"Failed to create training data loader: {e}")
        raise e
    
    print("Creating test data loader...")
    try:
        test_loader = build_data_loader(config=test_config.data,exp_config = exp_config)
        print("Test data loader ready.")
    except Exception as e:
        print(f"Failed to create test data loader: {e}")
        raise e

    criterion = soft_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)

    scheduler = None
    if exp_config.get('use_cos_schedule',False):
        assert train_config.use_steps == True  
        print("USING COS SCHEDULER LR")                
        warmup_steps = max(1, int(0.05 * train_config.num_steps))   # 5 % warm‑up
        total_steps = train_config.num_steps
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:       
                return current_step / warmup_steps
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    init_loss = eval_epoch(model, criterion, test_loader, device)
    print(f"[{exp_config['exp_id']} | Initial Test Loss: {init_loss:.4f}")

    # Optional initial puzzle evaluation before training starts.
    initial_overall_acc = 0.0
    if exp_config.get('eval_initial_puzzle', True):
        print("Running initial puzzle evaluation...")
        try:
            initial_puzzle_result = eval_DPU_on_puzzles(
                out_path, model=model, save_name_suffix='initial',
                chess_data_root=exp_config.get('chess_data_root'),
            )
            from src.evaluate.plotting import plot_bar
            _, _, initial_overall_acc = plot_bar(initial_puzzle_result, no_plot=True)
            print(f"Initial puzzle accuracy: {initial_overall_acc:.3f}")
        except Exception as e:
            print(f"Initial puzzle evaluation failed: {e}")
            initial_overall_acc = 0.0
    else:
        print("Skipping initial puzzle evaluation.")

    best_test_loss = init_loss
    best_test_model_path = os.path.join(out_path, 'best_test_model.pth')
    if train_config.use_steps:
      puzzle_steps_eval = exp_config.get('puzzle_steps_eval',None)

      results = {
          "step_num":[],
          "step_train_loss": [],
          "step_test_loss": [],
          "init_test_loss": init_loss,
          "init_puzzle_acc": initial_overall_acc,
          "params_ct":  params_ct
      }
      if puzzle_steps_eval is not None:
          results['puzzle_acc'] = []
          results['puzzle_step_num'] = []

      model.train()
      step, epoch = 1, 1
      data_iter = iter(train_loader)

      while step <= train_config.num_steps:
          interval_steps = min(train_config.num_steps_eval, train_config.num_steps - step + 1)
          train_loss, data_iter, step, epoch = train_steps(model, optimizer, criterion, train_loader, data_iter, device, interval_steps, step, epoch, scheduler)

          results["step_num"].append(step)
          results["step_train_loss"].append(train_loss)

          test_loss = eval_epoch(model, criterion, test_loader, device)
          results["step_test_loss"].append(test_loss)

          if test_loss < best_test_loss:
            best_test_loss = test_loss
            print(f"Best Test Loss Model saved at Step = {step}")

            checkpoint = {
            "model": model,
            "config": {**exp_config,**{'checkpoint_model_step': step}}
            }
            torch.save(checkpoint, best_test_model_path)

          if (puzzle_steps_eval is not None) and ((step-1) % puzzle_steps_eval == 0):
              result_df = eval_DPU_on_puzzles(
                  out_path, model=model,
                  chess_data_root=exp_config.get('chess_data_root'),
              )
              _, _, overall_acc = plot_bar(result_df,no_plot = True)
              results['puzzle_acc'].append(overall_acc)
              results['puzzle_step_num'].append(step)
              training_puzzle_plotting(results['puzzle_step_num'], 'Step', results['puzzle_acc'], out_path)

          training_loss_plotting(results["step_num"],'Step',results["step_test_loss"],results["step_train_loss"],out_path)
          current_lr = optimizer.param_groups[0]['lr']
          print(
              f"[{exp_config['exp_id']}|Step {step}] "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"LR: {current_lr:.6e}"
          )

    else: # train by epoch
      results = {
          "epoch_train_loss": [],
          "epoch_test_loss": [],
          "init_test_loss": init_loss,
          "init_puzzle_acc": initial_overall_acc,
          "params_ct": params_ct
      }

      for epoch in range(train_config.num_epoch):
          train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
          results["epoch_train_loss"].append(train_loss)

          test_loss = eval_epoch(model, criterion, test_loader, device)
          results["epoch_test_loss"].append(test_loss)

          if test_loss < best_test_loss:
            best_test_loss = test_loss
            print(f"Best Test Loss Model saved at Epoch = {epoch + 1}")
            checkpoint = {
            "model": model,
            "config": {**exp_config,**{'Model_Epoch': epoch+ 1}}
            }
            torch.save(checkpoint, best_test_model_path)

          training_loss_plotting(None,'Epoch',results["epoch_test_loss"],results["epoch_train_loss"],out_path)
          print(  
              f"[{exp_config['exp_id']}|Epoch {epoch + 1}] "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}"
          )

    identifier = str(ensemble) if ensemble else ''

    model_path = os.path.join(out_path, f'model{identifier}.pth')
    checkpoint = {
      "model": model,
      "config": exp_config
    }
    torch.save(checkpoint, model_path)

    # save loss records
    with open(os.path.join(out_path, 'record.pkl'), "wb") as f:
      pickle.dump(results, f)

    print(f"[{exp_config['exp_id']} Done. Results saved to {out_path}")
    return out_path



