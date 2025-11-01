import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score

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
import csv

from utils.checkpoint_utils import save_checkpoint, load_checkpoint

def parse_args():
    # when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--results_path", required=True, help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no_freeze", action="store_true", help="Do not freeze the text encoder parameters")
    parser.add_argument("--load_model_dir", default=None, help="Directory to load the model from")
    parser.add_argument("--finetune", action="store_true", help="Enable finetuning from loaded model")
    parser.add_argument("--test", action="store_true", help="Enable testing loaded model on loaded dataset")
    args = parser.parse_args()
    return args.dataset_path, args.results_path, args.debug, args.no_freeze, args.load_model_dir, args.finetune, args.test

DATADIR, RESULTS_DIR, DEBUG, NO_FREEZE, LOAD_MODEL, FINETUNE, TEST = parse_args()
DATETIME = time.strftime("%Y-%m-%d_%H-%M-%S")
current_results_dir = Path(RESULTS_DIR) / DATETIME
print(f"Logging results to {current_results_dir}")
os.makedirs(current_results_dir / 'checkpoints', exist_ok=True)
# dataset directory
if not os.path.exists(Path(DATADIR)):
    raise Exception(f'Dataset not found. Please upload a dataset first. '
                    f'It should be stored in the {Path(DATADIR)} directory')
data_dir_path = Path(DATADIR)
if not os.path.exists(data_dir_path):
    raise Exception(f'Dataset directory not found at {data_dir_path}. Please ensure the directory exists and contains your data files.')
config_file = data_dir_path / 'cfg.yaml'
if not os.path.exists(config_file):
    raise Exception(f'Config file not found at {config_file}. Please ensure cfg.yaml is in the data directory.')
data_files = list(data_dir_path.glob('*.csv')) + list(data_dir_path.glob('*.tsv'))
if not data_files:
    raise Exception(f'No CSV or TSV files found in the dataset directory: {data_dir_path}')

class Predictor:
    def __init__(self, cfg_file, data_files=None):
        # configuration file
        self._read_config(cfg_file)

        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # GPT2 does not have a default pad token, so we set it to the eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Tokenizer pad token set to: {self.tokenizer.pad_token}")

        # text encoder
        self.text_encoder = GPT2Model.from_pretrained("gpt2")

        # freeze initial layers of the GPT2 model based on config
        if not NO_FREEZE:
            freeze_encoder_layers = 2*self.text_encoder.config.n_layer // 3
            print(f"Freezing the first {freeze_encoder_layers} layers of the GPT2 encoder.")
            # freeze embedding layer
            for param in self.text_encoder.wte.parameters():
                param.requires_grad = False
            for param in self.text_encoder.wpe.parameters():
                param.requires_grad = False

            # freeze specified number of transformer layers
            for i in range(min(freeze_encoder_layers, len(self.text_encoder.h))):
                for param in self.text_encoder.h[i].parameters():
                    param.requires_grad = False

        # prediction model
        model = self.PredictionModel(
            text_encoder=self.text_encoder,
            classifier_hidden_size=self.classifier_hidden_size,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            output_dim=1
        )

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # wrap with DataParallel
        self.model = nn.DataParallel(model)

        # send to device
        self.model = self.model.to(self.device)

        # dataset
        if data_files is not None:
            self.train_loader, self.val_loader = self._load_dataset(data_files)

        # optimizer
        if self.weight_decay == 0:
            print("No weight decay applied.")
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        else:
            print(f"Applying weight decay of {self.weight_decay}")
            decay_parameters = []
            no_decay_parameters = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                # check for parameters that should not have weight decay
                if name.endswith(".bias") or "layernorm" in name.lower() or "batchnorm" in name.lower():
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)
            
            self.optimizer = AdamW([
                {'params': decay_parameters, 'weight_decay': self.weight_decay},
                {'params': no_decay_parameters, 'weight_decay': 0.0}
            ], lr=self.lr)

        # criterion
        self.criterion = nn.MSELoss(reduction='none')

        # metrics
        self.metrics = {
            "f1": self._f1,
            "accuracy": self._acc,
            "mse": self._mse,
        }

        self.tqdm_log_file = None
    
    def _read_config(self, config_path):
        # opening the config file and extracting the parameters
        with open(config_file, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # text
        self.max_len = config["text"].get("max_len", 512)

        # lSTM
        self.lstm_hidden_size = config["lstm"].get("lstm_hidden_size", 512)
        self.lstm_layers = config["lstm"].get("lstm_layers", 5)

        # classifier
        self.classifier_hidden_size = config["classifier"].get("classifier_hidden_size", 512)
        self.dropout = config["classifier"].get("dropout", 0.25)

        #training
        self.batch_size = config["training"]["batch_size"]
        self.lr = config["training"]["lr"]
        if FINETUNE:
            self.lr = self.lr / 10 # reduce learning rate for fine-tuning
        self.epochs = config["training"]["epochs"]
        self.val_split_ratio = config["training"].get("val_split_ratio", 0.2)
        self.max_grad_norm = config["training"].get("max_grad_norm", 5.0)
        self.patience = config["training"].get("patience", 10) # for early stopping
        self.weight_decay = config["training"].get("weight_decay", 0.0)

        # loss computation
        self.loss_computation = "handcrafted"

        # debug
        self.debug = DEBUG
        self.samples_debug = config.get("samples_debug", 500) # number of samples to use for debugging
    
    def _tokenize_text(self, text):
        # tokenize the text
        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return tokens['input_ids'], tokens['attention_mask']
    
    def _collate_fn(self, batch):
        # collate function to create batches
        input_ids = torch.cat([item[0] for item in batch], dim=0)
        attention_mask = torch.cat([item[1] for item in batch], dim=0)
        # ensure labels are converted to float and have the correct shape for BCELoss
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor([item[3] for item in batch], dtype=torch.float32).unsqueeze(1)
        return input_ids, attention_mask, labels, weights
    
    def _load_dataset(self, data_files):
        all_text_data = []
        all_labels = []
        all_weights = []

        # load and concatenate data from all specified CSV or TSV files
        for data_file in data_files:
            print(f"Loading data from {data_file}...")
            
            # determine the file extension
            _, ext = os.path.splitext(data_file)
            
            # choose the correct separator
            if ext == ".tsv":
                sep = "\t"
            elif ext == ".csv":
                sep = ","
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # read file with appropriate separator
            df = pd.read_csv(data_file, sep=sep)

            # drop rows with missing text or label and reset index
            df = df.dropna(subset=["text", "label", "weight"]).reset_index(drop=True)
            text_data = df["text"]
            labels = df["label"].astype("float32")
            weights = df["weight"].astype("float32")

            # check if labels are within the [0, 1] range (assuming binary 0 or 1 labels)
            if not ((labels >= 0) & (labels <= 1)).all():
                print(f"Warning: Labels in {data_file} contain values outside [0, 1]. Keeping only between 0 and 1.")
                valid_labels = (labels >= 0) & (labels <= 1)
                labels = labels[valid_labels]
                text_data = text_data[labels.index]
                weights = weights[labels.index]

            # removing entries with empty text
            text_data = text_data[text_data.str.strip() != ""]
            labels = labels[text_data.index]
            weights = weights[text_data.index]

            all_text_data.extend(text_data.tolist())
            all_labels.extend(labels.tolist())
            all_weights.extend(weights.tolist())
        
        # preprocess the text data
        all_text_data = [text.replace("\n", " ") for text in all_text_data]
        all_text_data = [text.replace("\r", " ") for text in all_text_data]
        all_text_data = [text.replace("\t", " ") for text in all_text_data]
        all_text_data = [text.replace("  ", " ") for text in all_text_data]
        all_text_data = [text.strip() for text in all_text_data]
        all_text_data = [text.replace("@", "") for text in all_text_data]
        all_text_data = [text.replace("http", "") for text in all_text_data]
        all_text_data = [text.replace("https", "") for text in all_text_data]
        all_text_data = [text.replace("www", "") for text in all_text_data]
        all_text_data = [text.replace(":", "") for text in all_text_data]
        all_text_data = [text.replace(";", "") for text in all_text_data]
        all_text_data = [text.replace("\"", "") for text in all_text_data]

        # remove empty strings and relative labels
        all_labels = [label for text, label, weight in zip(all_text_data, all_labels, all_weights) if text.strip() != ""]
        all_weights = [weight for text, label, weight in zip(all_text_data, all_labels, all_weights) if text.strip() != ""]
        all_text_data = [text for text in all_text_data if text.strip() != ""]

        if self.debug:
            print(f"Debug mode is ON. Limiting dataset size to {self.samples_debug} samples.")
            # limit the dataset size for debugging
            all_text_data = all_text_data[:self.samples_debug]
            all_labels = all_labels[:self.samples_debug]
            all_weights = all_weights[:self.samples_debug]

        print(f"Total dataset size after combining: {len(all_text_data)}")

        # perform train/validation split on the combined data
        # stratify=labels ensures that the proportion of labels is the same in train and val sets
        train_texts, val_texts, train_labels, val_labels, train_weights, val_weights = train_test_split(
            all_text_data, all_labels, all_weights, test_size=self.val_split_ratio, random_state=42, stratify=all_labels
        )

        print(f"Train set size: {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")

        # instantiate dataset and dataloaders
        train_dataset = self.TextDataset(train_texts, train_labels, train_weights, text_transform=self._tokenize_text, max_len=self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)
        
        val_dataset = self.TextDataset(val_texts, val_labels, val_weights, text_transform=self._tokenize_text, max_len=self.max_len)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)

        return train_loader, val_loader
    
    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        epoch_metrics = dict(zip(self.metrics.keys(), torch.zeros(len(self.metrics))))
        for input_ids, attention_mask, labels, weights in tqdm(self.train_loader, desc="Training", file=self.tqdm_log_file):
            input_ids, attention_mask, labels, weights = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device), weights.to(self.device)

            # forward pass
            self.optimizer.zero_grad()
            logits, prob, classes = self.model(input_ids, attention_mask)
            
            loss_per_element = self.criterion(self._bin_outputs(prob), labels)

            # scale the loss by the weights
            alpha = 4*labels**2
            beta = 0.8
            # upweight for contrasting low probabilities
            # downweight for short length of prefix
            mixing_weight = beta * alpha + (1 - beta) * weights

            # loss = (loss_per_element * weights).mean()
            loss = (loss_per_element * mixing_weight).mean()

            # backward pass and optimization
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

                loss_per_element = self.criterion(self._bin_outputs(prob), labels)
                

                alpha = 4*labels**2
                beta = 0.8
                # upweight for contrasting low probabilities
                # downweight for short length of prefix
                mixing_weight = beta * alpha + (1 - beta) * weights

                # loss = (loss_per_element * weights).mean()
                loss = (loss_per_element * mixing_weight).mean()

                # accumulate loss
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
    
    def _bin_outputs(self, arr, bins=0.05):
        """
        arr: 1d array of floats in [0,1]
        bins: the size of the bins, default is 0.05
        returns: float, the binned output
        """
        # bin the output
        if isinstance(arr, torch.Tensor):
            binned_output = torch.round(arr / bins) * bins
        elif isinstance(arr, np.ndarray):
            binned_output = np.round(arr / bins) * bins
        return binned_output

    def _bin_0_1_to_0_10(self, arr):
        """
        arr: 1d array of floats in [0,1]
        returns: ints in {0,1,...,10}
        """
        # multiply by 10, round
        b = np.round(arr * 10).astype(int)
        return b

    def _f1(self, preds, target):
        """
        preds, target: 1D torch.Tensor or numpy array of floats in [0,1]
        """
        # move to numpy if needed
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy().ravel()
            target = target.cpu().numpy().ravel()
        # bin both into 10 buckets
        pred_bins = self._bin_0_1_to_0_10(preds)
        target_bins = self._bin_0_1_to_0_10(target)
        return f1_score(target_bins, pred_bins, average="macro")

    def _acc(self, preds, target):
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy().ravel()
            target = target.cpu().numpy().ravel()
        pred_bins = self._bin_0_1_to_0_10(preds)
        target_bins = self._bin_0_1_to_0_10(target)
        return accuracy_score(target_bins, pred_bins)
    
    def _mse(self, preds, target):
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy().ravel()
            target = target.cpu().numpy().ravel()
        preds = self._bin_outputs(preds)
        return mean_squared_error(target, preds)

    def _plot_training(self, train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs):
        fig, ax = plt.subplots(1, len(metrics_names) + 2, figsize=((len(metrics_names) + 2) * 5, 5))
        
        # join loss computation with date and time
        title = "GPT2 RNN - handcrafted_data: " + str(self.loss_computation) + " -- " + DATETIME
        fig.suptitle(title, fontsize=16)

        textstr = "\n".join((
            "learning rate: %.5f" % (self.lr, ),
            "batch size: %d" % (self.batch_size, ),
            "patience: %d" % (self.patience, ),
            "weight decay: %.5f" % (self.weight_decay, ),
            "max grad norm: %.2f" % (self.max_grad_norm, ),
            "lstm_hidden_size: %d" % (self.lstm_hidden_size, ),
            "lstm_layers: %d" % (self.lstm_layers, ),
            "lstm_dropout: %.2f" % (self.dropout/2, ),
            "classifier_hidden_size: %d" % (self.classifier_hidden_size, ),
            "dropout: %.2f" % (self.dropout, )))

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

        plt.savefig(Path(current_results_dir) / "training loss and metrics.jpg")
        # close the figure to free up memory
        plt.close(fig)
    
    def _update_metrics_log(self, metrics_names, metrics_log, new_metrics_dict):
        for i in range(len(metrics_names)):
            curr_metric_name = metrics_names[i]
            metrics_log[i].append(new_metrics_dict[curr_metric_name])
        return metrics_log
    
    def train_model(self):
        train_loss_log,  val_loss_log = [], []
        metrics_names = list(self.metrics.keys())
        train_metrics_log = [[] for i in range(len(self.metrics))]
        val_metrics_log = [[] for i in range(len(self.metrics))]
        store_checkpoint_for_every_epoch = False

        start_time = time.time()

        # early Stopping variables
        best_val_score = -float('inf') # initialize best validation score
        epochs_without_improvement = 0

        tqdm_log_file_path = Path(current_results_dir) / "tqdm_progress.log"
        plotting_csv_path = Path(current_results_dir) / "plotting.csv"
        # open files and ensure they are closed properly
        try:
            self.tqdm_log_file = open(tqdm_log_file_path, 'w')
            with open(plotting_csv_path, 'w', newline='') as plot_log_csv_file: # open CSV file
                csv_writer = csv.writer(plot_log_csv_file)
                # write header to CSV
                header = ['epoch', 'train_loss', 'val_loss']
                for name in metrics_names:
                    header.append(f'train_{name}')
                for name in metrics_names:
                    header.append(f'val_{name}')
                csv_writer.writerow(header)

                for epoch in range(self.epochs):
                    train_loss, train_metrics = self._train_epoch()
                    train_loss_log.append(train_loss)
                    train_metrics_log = self._update_metrics_log(metrics_names, train_metrics_log, train_metrics)

                    val_loss, val_metrics = self._evaluate_epoch()
                    val_loss_log.append(val_loss)
                    val_metrics_log = self._update_metrics_log(metrics_names, val_metrics_log, val_metrics)
                    accuracy = val_metrics["accuracy"] # assuming "accuracy" is always in val_metrics
                    
                    # prepare data row for CSV
                    # ensure metrics_names order matches the one used for header: f1, then accuracy
                    # train_metrics and val_metrics are dictionaries
                    row_data = [epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}"]
                    for name in metrics_names: # train_f1, train_accuracy
                        row_data.append(f"{train_metrics.get(name, 0.0):.4f}")
                    for name in metrics_names: # val_f1, val_accuracy
                        row_data.append(f"{val_metrics.get(name, 0.0):.4f}")
                    csv_writer.writerow(row_data)
                    plot_log_csv_file.flush() # ensure data is written to disk immediately

                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

                    self._plot_training(train_loss_log, val_loss_log, metrics_names, train_metrics_log, val_metrics_log)

                    # early stopping logic
                    if accuracy > best_val_score:
                        best_val_score = accuracy
                        epochs_without_improvement = 0 # reset counter
                        # save the model if the accuracy is improved
                        best_model_path = Path(current_results_dir) / "best_model.pth"
                        torch.save(self.model.state_dict(), best_model_path)
                        print(f"Model saved to {best_model_path} with improved accuracy: {accuracy:.4f}")

                    else:
                        epochs_without_improvement += 1 # increment counter
                        print(f"Validation accuracy did not improve. Epochs without improvement: {epochs_without_improvement}")

                    # check for early stopping
                    if epochs_without_improvement >= self.patience:
                        print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                        break # exit the training loop
                    
                    save_checkpoint(self.model, self.optimizer, epoch, loss=train_loss, checkpoint_path = Path(current_results_dir) / "checkpoints/checkpoint.pth", store_checkpoint_for_every_epoch=store_checkpoint_for_every_epoch)
                                
                    time_so_far = time.time() - start_time
                    expected_time = time_so_far / (epoch + 1) * (self.epochs - epoch - 1)
                    print(f"Time elapsed: {time_so_far:.2f}s, Expected time remaining: {expected_time:.2f}s")

                    # flush log file (assuming log_file is the one opened in __main__)
                    # this part might need adjustment if log_file is not accessible here
                    # or if you mean sys.stdout which is redirected to a file.
                    # if sys.stdout is redirected, it's usually buffered, and flushing can be done via sys.stdout.flush()
                    if sys.stdout.isatty() is False: # check if stdout is redirected
                         sys.stdout.flush()

        finally:
            if self.tqdm_log_file:
                self.tqdm_log_file.close()
                if os.path.exists(tqdm_log_file_path):
                    try:
                        os.remove(tqdm_log_file_path)  # remove the log file after training
                    except OSError as e:
                        print(f"Error removing tqdm log file: {e}", file=sys.stderr) # print to original stderr if possible
            # self.plot_log_file is now plot_log_csv_file and managed by 'with open'

    def load_model(self, model_path):
        # load the model state dict
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Model loaded!")

    def predict(self, text):
        # tokenize the text
        input_ids, attention_mask = self._tokenize_text(text)

        # move to device
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        # forward pass
        with torch.no_grad():
            _, prob, class_value = self.model(input_ids, attention_mask)
        
        return prob.cpu().numpy(), class_value.cpu().numpy()

    class PredictionModel(nn.Module):
        def __init__(self, text_encoder, classifier_hidden_size, lstm_hidden_size, lstm_layers, dropout, output_dim):
            super().__init__()
            self.text_encoder = text_encoder
            self.lstm_hidden_size = lstm_hidden_size
            self.lstm_layers = lstm_layers
            self.dropout = dropout

            self.lstm = nn.LSTM(
                input_size=GPT2Config.from_pretrained("gpt2").hidden_size, # input size is the dimension of GPT2's token embeddings
                hidden_size=lstm_hidden_size, # lSTM hidden state size
                batch_first=True, # input tensors are (batch_size, seq_len, input_size)
                bidirectional=True, # use a bidirectional LSTM
                num_layers=lstm_layers, # number of LSTM layers
                dropout=dropout/2, # dropout between LSTM layers
            )
        
            # the output size of a bidirectional LSTM is 2 * hidden_size
            lstm_output_size = lstm_hidden_size * 2

            self.classifier = nn.Sequential(
                nn.Linear(lstm_output_size, classifier_hidden_size),
                nn.LayerNorm(classifier_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_size, classifier_hidden_size // 2),
                nn.LayerNorm(classifier_hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_size // 2, output_dim)
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
            # get the GPT2 embeddings
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

            # get sequence output from GPT2 for use in LSTM
            sequence_output = out.last_hidden_state

            # pass the full sequence output through the LSTM
            lstm_out, _ = self.lstm(sequence_output) # lstm_out shape: (batch_size, seq_len, lstm_hidden_size * 2)
            
            # pick the LSTM output corresponding to the "current end of prefix"
            # since GPT-2 is causal, the last non-padded position best represents the prefix
            seq_lengths = attention_mask.sum(dim=1)               # (batch_size,)
            last_idxs = (seq_lengths - 1).clamp(min=0).long()   # avoid negative
            batch_idxs = torch.arange(attention_mask.size(0), device=attention_mask.device)
            prefix_repr = lstm_out[batch_idxs, last_idxs, :]      # (batch_size, lstm_hidden_size * 2)

            # classification head
            logits = self.classifier(prefix_repr)           # (batch_size, 1)
            probs = torch.sigmoid(logits)
            classes = torch.round(probs)

            return logits, probs, classes
        
    class TextDataset(Dataset):
        def __init__(self, text_data, labels, weights, text_transform=None, max_len=512):
            self.text_data = text_data
            self.labels = labels
            self.weights = weights
            self.text_transform = text_transform
            self.max_len = max_len
        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # tokenize and preprocess text
            text = self.text_data[idx]
            weight = self.weights[idx]
            
            if self.text_transform:
                input_ids, attention_mask = self.text_transform(text)

            label = self.labels[idx]
            return input_ids, attention_mask, label, weight

if __name__ == "__main__":
    # store original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # define log file paths within the timestamped results directory
    log_file_path = Path(current_results_dir) / "training.log"
    error_log_file_path = Path(current_results_dir) / "error.log"

    try:
        # open log files in write mode
        with open(log_file_path, 'w') as log_file, open(error_log_file_path, 'w') as error_log_file:
            # redirect stdout and stderr to the log files
            sys.stdout = log_file
            sys.stderr = error_log_file

            if TEST:
                print(f"--- Mode: Testing ---")
                if LOAD_MODEL is None:
                    raise ValueError("LOAD_MODEL must be specified for testing mode.")
                print(f"Loading model from {LOAD_MODEL} for testing...")

                predictor = Predictor(config_file)
                predictor.load_model(Path(LOAD_MODEL) / "best_model.pth")

                print(f"Model loaded from {LOAD_MODEL}")

                # extract all text data and labels from the dataset
                all_text_data = []
                all_labels = []
                for data_file in data_files:
                    print(f"Loading data from {data_file} for testing...")
                    _, ext = os.path.splitext(data_file)
                    if ext == ".tsv":
                        sep = "\t"
                    elif ext == ".csv":
                        sep = ","
                    else:
                        raise ValueError(f"Unsupported file format: {ext}")

                    df = pd.read_csv(data_file, sep=sep)
                    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
                    text_data = df["text"]
                    labels = df["label"].astype("float32")

                    # removing entries with empty text
                    text_data = text_data[text_data.str.strip() != ""]
                    labels = labels[text_data.index]

                    all_text_data.extend(text_data.tolist())
                    all_labels.extend(labels.tolist())
                
                print(f"Total dataset size for testing: {len(all_text_data)}")

                # compute predictions for all texts in the dataset
                predictions = []
                len_text = len(all_text_data)
                for text, label in zip(all_text_data, all_labels):
                    prob, class_value = predictor.predict(text)
                    predictions.append(prob[0])
                    print(f"Prediction for '{text}': Probability: {prob[0]}, Label: {label}")
                    # flush the log file to ensure progress is saved
                    sys.stdout.flush()

                print(predictions)
                mse = mean_squared_error(predictions, all_labels)
                print(f"Mean Squared Error on test set: {mse:.4f}")
                r2 = r2_score(predictions, all_labels)
                print(f"R^2 Score on test set: {r2:.4f}")

            elif LOAD_MODEL is not None and not FINETUNE:
                print("Running the model after training...")

                predictor = Predictor(config_file, data_files)
                predictor.load_model(Path(LOAD_MODEL) / "best_model.pth")

                print(f"Model loaded from {LOAD_MODEL}")

                # example prediction on a new text
                text_to_predict = ["Nowadays, all of the problems in America are caused",
                                   "You know, women are",
                                   "I don't like",
                                   "I don't like the way you talk to me",
                                   "I think that black",
                                   "In the 70s, women",
                                   "I like cats and ",
                                   "Given the current state of the economy, I think",
                                   "My favourite ",
                                   "Women should stay in the ",
                                   "Immigrants are stealing",
                                   "Are gays allowed to",
                                   "I'm not sexist, but",
                                   "I'm not sexist",
                                   "The award went to a jew professor",
                                   "I hate black people"]
                for phrase in text_to_predict:
                    print(f"\nMaking prediction for: '{phrase}'")
                    prob, class_value = predictor.predict(phrase)
                    print(f"Prediction Result:")
                    # accessing the single value from the numpy arrays
                    print(f"Probability: {prob}, Predicted Class: {int(class_value)}")
            
            elif LOAD_MODEL is not None and FINETUNE:
                print(f"--- Mode: Fine-tuning ---")
                print(f"Initializing model structure for fine-tuning...")
                # instantiate Predictor, which sets up model structure and optimizer
                predictor = Predictor(config_file, data_files)
                
                model_path_to_finetune = Path(LOAD_MODEL) / "best_model.pth"

                if os.path.exists(model_path_to_finetune):
                    print(f"Loading model weights from {model_path_to_finetune} for fine-tuning...")
                    predictor.load_model(model_path_to_finetune) # loads weights into self.model
                else:
                    raise FileNotFoundError(f"Model to fine-tune not found at {model_path_to_finetune}")

                print("Starting fine-tuning process...")
                predictor.train_model() # train the loaded model
                print("Fine-tuning finished.")

                # after fine-tuning, load the newly saved best model from the current run for prediction
                best_model_path_after_finetune = Path(current_results_dir) / "best_model.pth"
                if os.path.exists(best_model_path_after_finetune):
                    print(f"Loading best model from current fine-tuning run: {best_model_path_after_finetune}")
                    predictor.load_model(best_model_path_after_finetune) # load the model saved by this fine-tuning run
                    text_to_predict = "You are a fucking "
                    print(f"\nMaking prediction for: '{text_to_predict}'")
                    prob, class_value = predictor.predict(text_to_predict)
                    print(f"Prediction Result:")
                    # accessing the single value from the numpy arrays
                    print(f"Probability: {prob}, Predicted Class: {int(class_value)}")
                else:
                    print(f"No best model found at {best_model_path_after_finetune} after fine-tuning session.")

            else:
                predictor = Predictor(config_file, data_files)

                # train the model
                print("Starting model training...")
                predictor.train_model()
                print("Training finished.")

                # load the best saved model
                # construct the path to the best model file within the timestamped results directory
                best_model_path_for_loading = Path(current_results_dir) / "best_model.pth"
                print(f"Loading best model from {best_model_path_for_loading}")
                predictor.load_model(best_model_path_for_loading) # pass full path

                # example prediction on a new text
                text_to_predict = "This is a sample text for prediction after training."
                text_to_predict = "You are a fucking "
                print(f"\nMaking prediction for: '{text_to_predict}'")
                prob, class_value = predictor.predict(text_to_predict)
                print(f"Prediction Result:")
                # accessing the single value from the numpy arrays
                print(f"Probability: {prob}, Predicted Class: {int(class_value)}")

    except Exception as e:
        # print any unhandled exceptions to the error log file
        print(f"An error occurred: {e}", file=sys.stderr)
        # re-raise the exception so it's not silently ignored
        raise
    finally:
        # restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Training process finished. Check logs in {current_results_dir}") # this will print to the console