import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import yaml
from tqdm import tqdm
import os
import sys
from pathlib import Path
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from utils.checkpoint_utils import save_checkpoint, load_checkpoint

import optuna

class Predictor:
    def __init__(self, cfg_file_path, data_files, hparams=None, trial_for_optuna=None):
        self.trial_for_optuna = trial_for_optuna
        self.trial_num = trial_for_optuna.number if trial_for_optuna else None

        # Store the main results directory. For HPO, subdirectories will be made within this.
        self.main_results_dir = current_results_dir # Uses the global current_results_dir

        self._read_config(cfg_file_path, hparams=hparams)
        
        # configuration file
        self._read_config(cfg_file_path)

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        # freeze all parameters except the pooler
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.pooler.parameters():
            param.requires_grad = True

        if not (hparams and hparams.get("is_hpo_setup_phase")): # Avoid printing during HPO setup if too verbose
            print(f"Text encoder model:\n{self.text_encoder}")

        # prediction model
        model = self.PredictionModel(
            text_encoder=self.text_encoder,
            hidden_size=self.hidden_size_classifier,
            output_dim=1,
            classifier_dropout=self.classifier_dropout # Pass tuned dropout
        )

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wrap with DataParallel
        self.model = nn.DataParallel(model)

        # Send to device
        self.model = self.model.to(self.device)

        # dataset
        self.train_loader, self.val_loader = self._load_dataset(data_files)

        # optimizer
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        # criterion
        if self.loss_computation == "classes" or self.loss_computation == "probabilities":
            self.criterion = nn.BCELoss(reduction='none')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        # metrics
        self.metrics = {
            "f1": self._f1,
            "accuracy": self._acc
        }

        self.tqdm_log_file = None
    
    def _read_config(self, config_path, hparams=None):
        # opening the config file and extracting the parameters
        with open(config_file, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # Base values from config
        base_lr = config["training"]["lr"]
        base_batch_size = config["training"]["batch_size"]
        base_hidden_size_classifier = config["text"].get("hidden_size", 512)
        base_classifier_dropout = 0.25 # Default from original PredictionModel

        # Hyperparameters from Optuna or defaults
        if hparams:
            self.lr = hparams.get("lr", base_lr)
            self.batch_size = hparams.get("batch_size", base_batch_size)
            self.hidden_size_classifier = hparams.get("hidden_size_classifier", base_hidden_size_classifier)
            self.classifier_dropout = hparams.get("classifier_dropout", base_classifier_dropout)
            # self.patience = hparams.get("patience", config["training"].get("patience", 10)) # Example if tuning patience
        else:
            self.lr = base_lr
            self.batch_size = base_batch_size
            self.hidden_size_classifier = base_hidden_size_classifier
            self.classifier_dropout = base_classifier_dropout
            # self.patience = config["training"].get("patience", 10)

        self.max_len = config["text"].get("max_len", 512)
        self.epochs = config["training"]["epochs"]
        self.val_split_ratio = config["training"].get("val_split_ratio", 0.2)
        self.max_grad_norm = config["training"].get("max_grad_norm", 5.0)
        self.patience = config["training"].get("patience", 10) # Keep patience from config for now, or tune it

        self.loss_computation = config.get("loss_computation", "logits")
        self.debug = DEBUG # Use global DEBUG
        self.samples_debug = config.get("samples_debug", 500)

    def _tokenize_text(self, text):
        # Tokenize the text
        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return tokens['input_ids'], tokens['attention_mask']
    
    def _collate_fn(self, batch):
        # Collate function to create batches
        input_ids = torch.cat([item[0] for item in batch], dim=0)
        attention_mask = torch.cat([item[1] for item in batch], dim=0)
        # Ensure labels are converted to float and have the correct shape for BCELoss
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor([item[3] for item in batch], dtype=torch.float32).unsqueeze(1)
        return input_ids, attention_mask, labels, weights
    
    def _load_dataset(self, data_files):
        all_text_data = []
        all_labels = []

        # Load and concatenate data from all specified CSV files
        for data_file in data_files:
            print(f"Loading data from {data_file}...")
            df = pd.read_csv(data_file)
            # Drop rows with missing text or label and reset index
            df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
            text_data = df["text"]
            labels = df["label"].astype("float32")

            # Check if labels are within the [0, 1] range (assuming binary 0 or 1 labels)
            if not ((labels >= 0) & (labels <= 1)).all():
                 print(f"Warning: Labels in {data_file} contain values outside [0, 1]. Keeping only 0 and 1.")
                 labels = labels[labels.isin([0, 1])]
                 text_data = text_data[labels.index]


            all_text_data.extend(text_data.tolist())
            all_labels.extend(labels.tolist())
        
        if self.debug:
            print(f"Debug mode is ON. Limiting dataset size to {self.samples_debug} samples.")
            # Limit the dataset size for debugging
            all_text_data = all_text_data[:self.samples_debug]
            all_labels = all_labels[:self.samples_debug]

        print(f"Total dataset size after combining: {len(all_text_data)}")

        # Perform train/validation split on the combined data
        # stratify=labels ensures that the proportion of labels is the same in train and val sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            all_text_data, all_labels, test_size=self.val_split_ratio, random_state=42, stratify=all_labels
        )

        print(f"Train set size: {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")

        # Instantiate dataset and dataloaders
        train_dataset = self.TextDataset(train_texts, train_labels, text_transform=self._tokenize_text, max_len=self.max_len, train=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)

        # weights = []
        # val_cut = []
        # # cut phrases in validation set only once
        # for text in val_texts:
        #     length = len(text.split())
        #     divider = np.random.randint(0, length)
        #     text = " ".join(text.split()[:divider]) # take the prefix
        #     val_cut.append(text)
        #     weight = divider / length
        #     weights.append(weight)
        
        val_dataset = self.TextDataset(val_texts, val_labels, text_transform=self._tokenize_text, max_len=self.max_len, train=False, weight=None)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)

        return train_loader, val_loader
    
    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        epoch_metrics = dict(zip(self.metrics.keys(), torch.zeros(len(self.metrics))))
        for input_ids, attention_mask, labels, weights in tqdm(self.train_loader, desc="Training", file=self.tqdm_log_file):
            input_ids, attention_mask, labels, weights = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device), weights.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, prob, classes = self.model(input_ids, attention_mask)
            if self.loss_computation == "classes":
                loss_per_element = self.criterion(classes, labels) # using classes computed from probabilities (BCELoss)
            elif self.loss_computation == "probabilities":
                loss_per_element = self.criterion(prob, labels)
            else:
                loss_per_element = self.criterion(logits, labels) # using logits directly (BCEWithLogitsLoss)
            # scale the loss by the weights

            loss = (loss_per_element * weights).mean()

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                for k in epoch_metrics.keys():
                    epoch_metrics[k] += self.metrics[k](classes.cpu(), labels.cpu())
        
        # empty tqdm log file
        self.tqdm_log_file.truncate(0)
        self.tqdm_log_file.seek(0)
        
        epoch_loss /= len(self.train_loader)
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(self.train_loader)

        # print('train Loss: {:.4f}, '.format(epoch_loss), ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

        return epoch_loss, epoch_metrics
    
    def _evaluate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        epoch_metrics = dict(zip(self.metrics.keys(), torch.zeros(len(self.metrics))))
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels, weights in tqdm(self.val_loader, desc="Evaluating", file=self.tqdm_log_file):
                input_ids, attention_mask, labels, weights = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device), weights.to(self.device)

                logits, prob, classes = self.model(input_ids, attention_mask)

                if self.loss_computation == "classes":
                    loss_per_element = self.criterion(classes, labels) # using classes computed from probabilities (BCELoss)
                elif self.loss_computation == "probabilities":
                    loss_per_element = self.criterion(prob, labels)
                else:
                    loss_per_element = self.criterion(logits, labels) # using logits directly (BCEWithLogitsLoss)
                
                # scale the loss by the weights
                loss = (loss_per_element * weights).mean()

                # Accumulate loss
                epoch_loss += loss.item()

                for k in epoch_metrics.keys():
                    epoch_metrics[k] += self.metrics[k](classes.cpu(), labels.cpu())
        
        # empty tqdm log file
        self.tqdm_log_file.truncate(0)
        self.tqdm_log_file.seek(0)

        epoch_loss /= len(self.val_loader)
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(self.val_loader)

        return epoch_loss, epoch_metrics
    
    def _f1(self, preds, target):
        return f1_score(target, preds, average='macro')

    def _acc(self, preds, target):
        return accuracy_score(target, preds)

    def _plot_training(self, train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs, results_dir_path_for_plot):
        fig, ax = plt.subplots(1, len(metrics_names) + 2, figsize=((len(metrics_names) + 2) * 5, 5))
        
        title_prefix = f"Trial {self.trial_num} - BERT" if self.trial_num is not None else "BERT"
        title = f"{title_prefix}: {self.loss_computation} -- {DATETIME}" # DATETIME is global
        fig.suptitle(title, fontsize=16)

        textstr = "\n".join((
            f"learning rate: {self.lr:.5f}",
            f"batch size: {self.batch_size:d}",
            f"patience: {self.patience:d}",
            f"classifier hidden: {self.hidden_size_classifier:d}",
            f"classifier dropout: {self.classifier_dropout:.2f}"
            ))

        ax[0].plot(train_loss, c='blue', label='train')
        ax[0].plot(test_loss, c='orange', label='validation')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].legend()

        for i in range(len(metrics_names)):
            ax[i + 1].plot(train_metrics_logs[i], c='blue', label='train')
            ax[i + 1].plot(test_metrics_logs[i], c='orange', label='validation')
            ax[i + 1].set_title(metrics_names[i])
            ax[i + 1].set_xlabel('epoch')
            ax[i + 1].legend()
        
        ax[-1].axis('off')
        ax[-1].text(0.5, 0.5, textstr, fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.savefig(Path(results_dir_path_for_plot) / "training loss and metrics.jpg")
        # close the figure to free up memory
        plt.close(fig)
    
    def _update_metrics_log(self, metrics_names, metrics_log, new_metrics_dict):
        for i in range(len(metrics_names)):
            curr_metric_name = metrics_names[i]
            metrics_log[i].append(new_metrics_dict[curr_metric_name])
        return metrics_log
    
    def train_model(self, is_hpo_trial=False):
        # Determine output directory for this specific training run
        if is_hpo_trial and self.trial_num is not None:
            # HPO trial: use a subdirectory within the main results directory
            output_dir = self.main_results_dir / f"optuna_trial_{self.trial_num}"
        else:
            # Single run or final HPO run: use the main results directory
            output_dir = self.main_results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        checkpoints_subdir = output_dir / 'checkpoints'
        os.makedirs(checkpoints_subdir, exist_ok=True)

        train_loss_log, val_loss_log = [], []
        metrics_names = list(self.metrics.keys())
        train_metrics_log = [[] for _ in range(len(self.metrics))]
        val_metrics_log = [[] for _ in range(len(self.metrics))]
        
        best_val_score = -float('inf')
        epochs_without_improvement = 0

        tqdm_log_file_path = output_dir / "tqdm_progress.log"
        try:
            self.tqdm_log_file = open(tqdm_log_file_path, 'w')

            for epoch in range(self.epochs):
                start_epoch_time = time.time()
                train_loss, train_metrics = self._train_epoch()
                train_loss_log.append(train_loss)
                train_metrics_log = self._update_metrics_log(metrics_names, train_metrics_log, train_metrics)

                val_loss, val_metrics = self._evaluate_epoch()
                val_loss_log.append(val_loss)
                val_metrics_log = self._update_metrics_log(metrics_names, val_metrics_log, val_metrics)
                accuracy = val_metrics["accuracy"]

                if not is_hpo_trial or (epoch % 5 == 0): # Reduce verbosity for HPO
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                if not is_hpo_trial: # Plotting only for non-HPO runs or final HPO run
                    self._plot_training(train_loss_log, val_loss_log, metrics_names, train_metrics_log, val_metrics_log, output_dir)

                if accuracy > best_val_score:
                    best_val_score = accuracy
                    epochs_without_improvement = 0
                    if not is_hpo_trial: # Save model only for non-HPO runs or final HPO run
                        best_model_path = output_dir / "best_model.pth"
                        torch.save(self.model.state_dict(), best_model_path)
                        print(f"Model saved to {best_model_path} with improved accuracy: {accuracy:.4f}")
                else:
                    epochs_without_improvement += 1
                    if not is_hpo_trial:
                         print(f"Validation accuracy did not improve. Epochs without improvement: {epochs_without_improvement}")


                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement at epoch {epoch+1}.")
                    break
                
                if is_hpo_trial and self.trial_for_optuna:
                    self.trial_for_optuna.report(accuracy, epoch)
                    if self.trial_for_optuna.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if not is_hpo_trial:
                    save_checkpoint(self.model, self.optimizer, epoch, loss=train_loss, checkpoint_path = checkpoints_subdir / "checkpoint.pth", store_checkpoint_for_every_epoch=False)
                    epoch_time_taken = time.time() - start_epoch_time
                    print(f"Epoch time: {epoch_time_taken:.2f}s")
        finally:
            if self.tqdm_log_file:
                self.tqdm_log_file.close()
            if os.path.exists(tqdm_log_file_path):
                 try: # Avoid error if file is already removed by another process/trial quickly
                    os.remove(tqdm_log_file_path)
                 except OSError:
                    pass
                    
        return best_val_score

    def load_model(self, model_path):
        # Load the model state dict
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Model loaded!")

    def predict(self, text):
        # Tokenize the text
        input_ids, attention_mask = self._tokenize_text(text)

        # Move to device
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        # Forward pass
        with torch.no_grad():
            _, prob, class_value = self.model(input_ids, attention_mask)
        
        return prob.cpu().numpy(), class_value.cpu().numpy()

    class PredictionModel(nn.Module):
        def __init__(self, text_encoder, hidden_size, output_dim, classifier_dropout=0.25):
            super().__init__()
            self.text_encoder = text_encoder

            self.classifier = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(hidden_size, output_dim)
            )
            self.classifier.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        def forward(self, input_ids, attention_mask):
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = out.pooler_output
            output = self.classifier(text_features)
            # print(f"Output: {output}")
            prob = torch.sigmoid(output)
            class_value = torch.round(prob)
            return output, prob, class_value
        
    class TextDataset(Dataset):
        def __init__(self, text_data, labels, text_transform=None, max_len=512, train=True, weight=None):
            self.text_data = text_data
            self.labels = labels
            self.text_transform = text_transform
            self.max_len = max_len
            self.train = train
            self.weight = weight

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # Tokenize and preprocess text
            text = self.text_data[idx]

            length = len(text.split())
            divider = np.random.randint(1, length+1)
            text = " ".join(text.split()[:divider]) # take the prefix
            weight = divider / length
            
            if self.text_transform:
                input_ids, attention_mask = self.text_transform(text)

            label = self.labels[idx]
            return input_ids, attention_mask, label, weight

# --- Optuna Objective Function ---
def objective(trial, cfg_path_obj, data_files_list_obj, base_hpo_results_dir_obj, global_debug_flag):
    # Hyperparameters to tune
    hparams = {
        "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]), # BERT is memory intensive
        "hidden_size_classifier": trial.suggest_categorical("hidden_size_classifier", [256, 512, 768]),
        "classifier_dropout": trial.suggest_float("classifier_dropout", 0.1, 0.5),
        # "patience": trial.suggest_int("patience", 5, 20), # Example: if you want to tune patience
        "is_hpo_setup_phase": True # To potentially suppress some prints during Predictor init
    }

    # Each trial will log its detailed output to a specific subdirectory
    # The 'base_hpo_results_dir_obj' is the main 'current_results_dir' for the HPO study
    trial_specific_log_dir = base_hpo_results_dir_obj / f"optuna_trial_{trial.number}"
    os.makedirs(trial_specific_log_dir, exist_ok=True)
    
    trial_stdout_path = trial_specific_log_dir / "trial_training.log"
    trial_stderr_path = trial_specific_log_dir / "trial_error.log"

    original_stdout_for_trial = sys.stdout
    original_stderr_for_trial = sys.stderr

    try:
        with open(trial_stdout_path, 'w') as trial_stdout_f, open(trial_stderr_path, 'w') as trial_stderr_f:
            sys.stdout = trial_stdout_f
            sys.stderr = trial_stderr_f

            print(f"--- Starting Optuna Trial {trial.number} ---")
            print(f"Hyperparameters: {hparams}")
            # DEBUG flag is global and will be picked up by Predictor's _read_config
            predictor = Predictor(cfg_path_obj, data_files_list_obj, hparams=hparams, trial_for_optuna=trial)
            best_val_accuracy = predictor.train_model(is_hpo_trial=True)
            print(f"Trial {trial.number} finished. Best Validation Accuracy: {best_val_accuracy:.4f}")
        
        # Restore stdout/stderr for this objective function's scope
        sys.stdout = original_stdout_for_trial
        sys.stderr = original_stderr_for_trial
        # Log completion to the main HPO study log (if redirected there) or console
        print(f"Optuna Trial {trial.number} completed. Best Val Accuracy: {best_val_accuracy:.4f}. Detailed logs in {trial_specific_log_dir}")
        return best_val_accuracy
        
    except optuna.exceptions.TrialPruned as e:
        sys.stdout = original_stdout_for_trial
        sys.stderr = original_stderr_for_trial
        print(f"Optuna Trial {trial.number} pruned: {e}")
        raise # Re-raise for Optuna to handle
    except Exception as e:
        sys.stdout = original_stdout_for_trial
        sys.stderr = original_stderr_for_trial
        # Log error to the main HPO study's error stream or console
        print(f"Optuna Trial {trial.number} FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return -1.0 # Return a poor value for failed trials



# --- Global variables to be set in main() ---
DATADIR = None
RESULTS_DIR = None
DEBUG = False
DATETIME = None
current_results_dir = None # This will be the main run's timestamped directory
config_file = None # Path object
data_files = None # List of Path objects


def main_script_logic():
    global DATADIR, RESULTS_DIR, DEBUG, DATETIME, current_results_dir, config_file, data_files

    # Argument Parsing (unified)
    parser = argparse.ArgumentParser(description="BERT Predictor with Optuna HPO.")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--results_path", required=True, help="Output directory for all runs/trials")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (uses fewer samples)")
    parser.add_argument("--hpo", action="store_true", help="Run Optuna hyperparameter optimization.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials if --hpo is set.")
    args = parser.parse_args()

    # Set global variables
    DATADIR = args.dataset_path
    RESULTS_DIR = args.results_path
    DEBUG = args.debug # This global DEBUG is used by Predictor

    DATETIME = time.strftime("%Y-%m-%d_%H-%M-%S")
    # current_results_dir is the main directory for this entire script execution
    current_results_dir = Path(RESULTS_DIR) / DATETIME 
    print(f"Main process results will be logged to: {current_results_dir}")
    os.makedirs(current_results_dir, exist_ok=True)
    
    # Setup paths for data and config (used by both HPO and single run)
    data_dir_path = Path(DATADIR)
    if not data_dir_path.exists():
        raise Exception(f'Dataset directory not found at {data_dir_path}.')
    config_file = data_dir_path / 'cfg.yaml'
    if not config_file.exists():
        raise Exception(f'Config file not found at {config_file}. Please ensure cfg.yaml is in the data directory.')
    data_files = list(data_dir_path.glob('*.csv'))
    if not data_files:
        raise Exception(f'No CSV files found in the dataset directory: {data_dir_path}')

    # Store original stdout/stderr for the main script process
    main_process_original_stdout = sys.stdout
    main_process_original_stderr = sys.stderr

    if args.hpo:
        print(f"\n--- Starting Optuna Hyperparameter Optimization ({args.n_trials} trials) ---")
        # Log Optuna's main study output to files within current_results_dir
        optuna_study_log_path = current_results_dir / "optuna_study_main.log"
        optuna_study_err_path = current_results_dir / "optuna_study_main_error.log"

        with open(optuna_study_log_path, 'w') as optuna_main_log_f, \
             open(optuna_study_err_path, 'w') as optuna_main_err_f:
            sys.stdout = optuna_main_log_f # Redirect Optuna's own verbose output
            sys.stderr = optuna_main_err_f

            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
            # Pass necessary (and now globally defined) paths and flags to the objective function
            study.optimize(lambda trial: objective(trial, config_file, data_files, current_results_dir, DEBUG),
                           n_trials=args.n_trials)

            # Restore stdout/stderr for the main process after Optuna study
            sys.stdout = main_process_original_stdout
            sys.stderr = main_process_original_stderr

        print("\n--- Optuna HPO Finished ---")
        print(f"Best trial achieved value: {study.best_trial.value:.4f}")
        print(f"Best hyperparameters: {study.best_trial.params}")
        
        study_results_df = study.trials_dataframe()
        study_results_df.to_csv(current_results_dir / "optuna_study_results.csv")
        print(f"Optuna study results saved to {current_results_dir / 'optuna_study_results.csv'}")

        print("\nTraining final model with best hyperparameters...")
        # The Predictor will use 'current_results_dir' for its output as trial_for_optuna will be None
        final_predictor = Predictor(config_file, data_files, hparams=study.best_trial.params)
        final_predictor.train_model(is_hpo_trial=False) # Full training run
        print(f"Final model training finished. Check logs and model in: {current_results_dir}")

    else: # Single training run (no HPO)
        print("\n--- Starting Single Model Training Run ---")
        # Log output of single run to files within current_results_dir
        single_run_log_path = current_results_dir / "training.log"
        single_run_err_path = current_results_dir / "error.log"
        
        # Ensure the checkpoints directory for the single run exists
        os.makedirs(current_results_dir / 'checkpoints', exist_ok=True)

        try:
            with open(single_run_log_path, 'w') as single_run_log_f, \
                 open(single_run_err_path, 'w') as single_run_err_f:
                sys.stdout = single_run_log_f
                sys.stderr = single_run_err_f

                predictor = Predictor(config_file, data_files) # Uses defaults from cfg.yaml
                predictor.train_model(is_hpo_trial=False)
                print("Single model training finished.")

                best_model_path_for_loading = current_results_dir / "best_model.pth"
                if best_model_path_for_loading.exists():
                    print(f"Loading best model from {best_model_path_for_loading}")
                    predictor.load_model(best_model_path_for_loading)
                    text_to_predict = "This is a sample text for prediction after training."
                    print(f"\nMaking prediction for: '{text_to_predict}'")
                    prob, class_value = predictor.predict(text_to_predict)
                    print(f"Prediction Result: Probability: {prob}, Predicted Class: {int(class_value)}")
                else:
                    print(f"Best model path {best_model_path_for_loading} not found. Skipping prediction.")
        except Exception as e:
            # Log to original stderr if file redirection failed or before it happened
            sys.stdout = main_process_original_stdout # Restore before printing error
            sys.stderr = main_process_original_stderr
            print(f"An error occurred during single run: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            # Restore stdout/stderr for the main process
            sys.stdout = main_process_original_stdout
            sys.stderr = main_process_original_stderr
            print(f"Single run process finished. Check logs in {current_results_dir}")


if __name__ == "__main__":
    # The parse_args() at the top is removed. All parsing happens in main_script_logic().
    # Global variables DATADIR, RESULTS_DIR, etc., are initialized within main_script_logic().
    main_script_logic()