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

def parse_args():
    # when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    return args.dataset_path, args.results_path

DATADIR, RESULTS_DIR = parse_args()
DATETIME = time.strftime("%Y-%m-%d_%H-%M-%S")
current_results_dir = Path(RESULTS_DIR) / DATETIME
print(f"Logging results to {current_results_dir}")
if not os.path.exists(current_results_dir):
    os.mkdir(current_results_dir)
# directory to store model checkpoints to
if not os.path.exists(Path(current_results_dir) / 'checkpoints'):
    os.mkdir(Path(current_results_dir) / 'checkpoints')
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
data_files = list(data_dir_path.glob('*.csv'))
if not data_files:
    raise Exception(f'No CSV files found in the dataset directory: {data_dir_path}')

class Predictor:
    def __init__(self, cfg_file, data_files):
        # configuration file
        self._read_config(cfg_file)

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        # freeze all parameters except the pooler
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.pooler.parameters():
            param.requires_grad = True

        # prediction model
        self.model = self.PredictionModel(
            text_encoder=self.text_encoder,
            hidden_size=self.hidden_size,
            output_dim=1
        )

        # dataset
        self.train_loader, self.val_loader = self._load_dataset(data_files)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
    
    def _read_config(self, config_path):
        # opening the config file and extracting the parameters
        with open(config_file, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # text
        self.max_len = config["text"].get("max_len", 512)
        self.hidden_size = config["text"].get("hidden_size", 512)

        #training
        self.batch_size = config["training"]["batch_size"]
        self.lr = config["training"]["lr"]
        self.epochs = config["training"]["epochs"]
        self.val_split_ratio = config["training"].get("val_split_ratio", 0.2)
        self.max_grad_norm = config["training"].get("max_grad_norm", 5.0)

        # loss computation
        self.loss_computation = config.get("loss_computation", "classes")
    
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

        print(f"Total dataset size after combining: {len(all_text_data)}")

        # Perform train/validation split on the combined data
        # stratify=labels ensures that the proportion of labels is the same in train and val sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            all_text_data, all_labels, test_size=self.val_split_ratio, random_state=42, stratify=all_labels
        )

        print(f"Train set size: {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")

        # Apply text augmentation only to the training set within the Dataset
        # Convert to lists after splitting
        train_texts_list  = train_texts
        val_texts_list    = val_texts
        train_labels_list = train_labels
        val_labels_list   = val_labels

        # Instantiate dataset and dataloaders
        train_dataset = self.TextDataset(train_texts, train_labels, text_transform=self._tokenize_text, max_len=self.max_len, train=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)

        val_dataset = self.TextDataset(val_texts, val_labels, text_transform=self._tokenize_text, max_len=self.max_len, train=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)

        return train_loader, val_loader
    
    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        epoch_metrics = dict(zip(self.metrics.keys(), torch.zeros(len(self.metrics))))
        for input_ids, attention_mask, labels, weights in tqdm(self.train_loader, desc="Training", file=self.tqdm_log_file):
            input_ids, attention_mask, labels, weights = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device), weights.to(self.device)

            # empty tqdm log file
            self.tqdm_log_file.truncate(0)
            self.tqdm_log_file.seek(0)

            # Forward pass
            self.optimizer.zero_grad()
            logits, prob, classes = self.model(input_ids, attention_mask)
            # print(f"Probabilities: {prob}")
            # print(f"Labels: {labels}")
            # Compute loss
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
        
        epoch_loss / len(self.train_loader)
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

                # empty tqdm log file
                self.tqdm_log_file.truncate(0)
                self.tqdm_log_file.seek(0)

                # Forward pass
                logits, prob, classes = self.model(input_ids, attention_mask)
                # Compute loss
                if self.loss_computation == "classes":
                    loss_per_element = self.criterion(classes, labels)
                elif self.loss_computation == "probabilities":
                    loss_per_element = self.criterion(prob, labels)
                else:
                    loss_per_element = self.criterion(logits, labels)
                # scale the loss by the weights
                loss= (loss_per_element * weights).mean()
                # Accumulate loss
                epoch_loss += loss.item()
                for k in epoch_metrics.keys():
                    epoch_metrics[k] += self.metrics[k](classes.cpu(), labels.cpu())
                
                # Compute accuracy
                # _, predicted = torch.max(classes, 1)
                correct += (classes == labels).sum().item()
                total += labels.size(0)

        epoch_loss /= len(self.val_loader)
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(self.val_loader)

        return epoch_loss, epoch_metrics
    
    def _f1(self, preds, target):
        return f1_score(target, preds, average='macro')

    def _acc(self, preds, target):
        return accuracy_score(target, preds)

    def _plot_training(self, train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs):
        fig, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))
        
        # join loss computation with date and time
        title = str(self.loss_computation) + " -- " + DATETIME
        fig.suptitle(title, fontsize=16)

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

        plt.savefig(Path(current_results_dir) / "training loss and metrics.jpg")
        # close the figure to free up memory
        plt.close(fig)
    
    def _update_metrics_log(self, metrics_names, metrics_log, new_metrics_dict):
        for i in range(len(metrics_names)):
            curr_metric_name = metrics_names[i]
            metrics_log[i].append(new_metrics_dict[curr_metric_name])
        return metrics_log
    
    def train_model(self):
        best_accuracy = 0
        train_loss_log,  val_loss_log = [], []
        metrics_names = list(self.metrics.keys())
        train_metrics_log = [[] for i in range(len(self.metrics))]
        val_metrics_log = [[] for i in range(len(self.metrics))]
        store_checkpoint_for_every_epoch = False

        tqdm_log_file_path = Path(current_results_dir) / "tqdm_progress.log"
        self.tqdm_log_file = open(tqdm_log_file_path, 'w')
        for epoch in range(self.epochs):
            train_loss, train_metrics = self._train_epoch()
            train_loss_log.append(train_loss)
            train_metrics_log = self._update_metrics_log(metrics_names, train_metrics_log, train_metrics)

            val_loss, val_metrics = self._evaluate_epoch()
            val_loss_log.append(val_loss)
            val_metrics_log = self._update_metrics_log(metrics_names, val_metrics_log, val_metrics)
            accuracy = val_metrics["accuracy"]
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

            self._plot_training(train_loss_log, val_loss_log, metrics_names, train_metrics_log, val_metrics_log)

            # Save the model if the accuracy is improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), "best_model.pth")
                print("Model saved!")

            # flush log file
            log_file.flush()
            
            save_checkpoint(self.model, self.optimizer, epoch, loss=train_loss, checkpoint_path = Path(current_results_dir) / "checkpoints/checkpoint.pth", store_checkpoint_for_every_epoch=store_checkpoint_for_every_epoch)

        os.remove(tqdm_log_file_path)  # Remove the log file after training

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
        def __init__(self, text_encoder, hidden_size, output_dim):
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
        def __init__(self, text_data, labels, train, text_transform=None, max_len=512):
            self.text_data = text_data
            self.labels = labels
            self.text_transform = text_transform
            self.max_len = max_len
            self.train = train

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # Tokenize and preprocess text
            text = self.text_data[idx]
            
            length = len(text.split())
            divider = np.random.randint(0, length)
            text = " ".join(text.split()[:divider]) # take the prefix
            weight = divider / length

            if self.text_transform:
                input_ids, attention_mask = self.text_transform(text)

            label = self.labels[idx]
            return input_ids, attention_mask, label, weight
#         def __getitem__(self, idx):

if __name__ == "__main__":
    # Store original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Define log file paths within the timestamped results directory
    log_file_path = Path(current_results_dir) / "training.log"
    error_log_file_path = Path(current_results_dir) / "error.log"

    try:
        # Open log files in write mode
        with open(log_file_path, 'w') as log_file, open(error_log_file_path, 'w') as error_log_file:
            # Redirect stdout and stderr to the log files
            sys.stdout = log_file
            sys.stderr = error_log_file

            # Now, all print statements and errors will go to these files

            # Entry point for the script
            # Instantiate the Predictor class
            # Pass the list of data files found in the data_dir
            predictor = Predictor(config_file, data_files)

            # Train the model
            print("Starting model training...")
            predictor.train_model()
            print("Training finished.")

            # Load the best saved model
            # Construct the path to the best model file within the timestamped results directory
            best_model_path_for_loading = Path(current_results_dir) / "best_model.pth"
            print(f"Loading best model from {best_model_path_for_loading}")
            predictor.load_model(best_model_path_for_loading) # Pass full path

            # Example prediction on a new text
            text_to_predict = "This is a sample text for prediction after training."
            print(f"\nMaking prediction for: '{text_to_predict}'")
            prob, class_value = predictor.predict(text_to_predict)
            print(f"Prediction Result:")
            # Accessing the single value from the numpy arrays
            print(f"Probability: {prob:.4f}, Predicted Class: {int(class_value)}")

    except Exception as e:
        # Print any unhandled exceptions to the error log file
        print(f"An error occurred: {e}", file=sys.stderr)
        # Re-raise the exception so it's not silently ignored
        raise
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Training process finished. Check logs in {current_results_dir}") # This will print to the console