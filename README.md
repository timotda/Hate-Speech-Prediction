# Anticipating Hate Speech From Partial Input
This project explores a novel approach to proactive hate speech moderation, aiming to predict the likelihood of hateful content as a sentence is being composed. This enables intervention before harmful messages are published, enhancing online safety. The report can be visualized [here](https://github.com/federock02/EPFL-EE559-HateSpeechDetection/blob/main/Report.pdf), please have a look.

Key Features:
- Real-time Probability Estimation: We develop models that output a continuous probability score for hate speech presence based on partial user input, rather than just binary (hate/non-hate) classification on complete texts.
- Hybrid Transformer Architectures: We investigate and compare both encoder-based (BERT) and decoder-based (GPT-2) transformer models, with optional recurrent (RNN) layers, to understand their strengths in this unique predictive task.
- Two-Stage Training Strategy: Models are first trained on large public binary-labeled hate speech datasets, then fine-tuned on a custom, manually annotated dataset of sentence prefixes with human-perceived hate probabilities.
- Weighted Loss for Nuance: A novel weighted loss function is introduced to prioritize early predictions and address label imbalance, particularly for less frequent, high-risk prefixes.

Our experiments reveal an intriguing trade-off:
- GPT-2 variants excelled in the initial binary classification phase on large datasets.
- BERT-based models delivered superior and more reliable probabilistic estimates on partial inputs, highlighting their strength in nuanced classification.
- The impact of RNN layers varied, sometimes improving performance by emphasizing sequential dependencies (e.g., in BERT's binary classification) and sometimes degrading the nuanced probability estimates.

## 1. Installing Requirements
To guarantee the presence of all the required libraries, this command should be run:
```bash
    pip install -r requirements.txt
```

## 2. Running the Python Script
You can launch the Python script using the command:
```bash
    python3 ~/Predictor/predictor_....py --<arguments>
```

All the arguments that can be added to the command are:
- `--dataset_path`: all the `.csv` files in the folder indicated in this argument will be used for the training and evaluation (required)
- `--result_path`: the logs, checkpoints, outputs and errors will be stored in the folder indicated by this argument (required)
- `--debug`: including this flag argument runs the training with a limited number of data samples, with the number indicated in the `cfg.yaml` in the dataset folder (optional)
- `--no_freeze`: flag to indicate that the training be be run with all the layers of the transformers in training mode, so it does not freeze the first pretrained layers (optional)
- `--load_model_dir`: indicates the path of the trained model that needs to be loaded, either for finetuning or for testing (optional)
- `--finetune`: flag that indicates that the model loaded will be finetuned (optional)
- `--test`: flag to activate the testing mode (optional)

### Base Training

The possible architectures that can be trained are:
- `peredictor_BERT.py`: uses *BERT* encoder as base text transformer
- `predictor_BERT_RNN.py`: uses *BERT* encoder as base text transformer, followed by a *biLSTM*
- `predictor_GPT2.py`: uses *GPT2* decoder as base text transformer
- `predictor_GPT2_RNN.py`: uses *GPT2* decoder as base text transformer, followed by a *biLSTM*

The hyperparameters used in the training process are expected to be in a `cfg.yaml` file, in the folder indicated as `--dataset_path`.

The dataset that are expected in this phase consist of classification datasets, labeled with binary labels. 1 is the positive class, 0 is the negative class. The phrases to train on are sourced from the `text` column in the `.csv` files, and will be cut in all the possible sub-prefixes of various length during the dataset loading phase; the labels are sourced from the `label` column.

### Finetuning

Once one of the previous models has been trained, it can be finetuned using a dataset labeled with probabilities. To finetune a trained model, it must be loaded by passing the path to the results folder for the previously trained model as argument after `--load_model_dir`, followed by the flag `--finetune`.

The phrases should again be in the `text` column of the `.csv` file, and this time they are expected to be prefixed of phrases, with the associated probability labels in the `label` column. There should also be a column `weight` indicating the percentage of words in the considered prefix compared to the length of the whole phrase.

To train each one of the models listed above, the corresponding `..._hadcrafted.py` model should be used, in order to guarantee that the trained weights are loaded in the right architecture for finetuning. The possible models are:
- `peredictor_BERT_handcrafted.py`: uses *BERT* encoder as base text transformer, considers probability labels
- `predictor_BERT_RNN_handcrafted.py`: uses *BERT* encoder as base text transformer, followed by a *biLSTM*, considers probability labels
- `predictor_GPT2_handcrafted.py`: uses *GPT2* decoder as base text transformer, considers probability labels
- `predictor_GPT2_RNN_handcrafted.py`: uses *GPT2* decoder as base text transformer, followed by a *biLSTM*, considers probability labels

All these models can also be traiend from scratch, if run without loading the model and without the finetuning flag.

## 3. Dataset preprocessing
We provide four useful script for managing the datasets:
- `utils/cut_phrases.py`: loads a dataset of complete phrases, cuts them into all the possible prefixes of various lengths, adds the original label relative to the complete sentence and associates the length percentage weight for each prefix.
- `utils/add_weights.py`: useful when a dataset with prefixes is available, and the weight for each prefix needs to be computed. Only works if to each prefix is associated an index that connects it to the original complete phrase.
- `utils/check_dataset.py`: returns some statistics about the dataset, and can be used for binary-labeled datasets of complete phrases.
- `utils/check_prefix_dataset.py`: returns some statistics about the dataset, and can be used for probability-labeled datasets of prefixes.

## 4. Testing
All the trained models can be loaded, using the same `peredictor_....py` it was trained with, adding the path of the trained weights after `--load_model_dir` and adding the `--test` flag. The data used for testing is loaded from the folder linked by `--dataset_path`.

## 5. Example Pipeline
Here follow an example of a training pipeline that can be executed, using the BERT-bsed model this time. This can easily be adapted for the other architectures.
1. Finding binary-labeled hate speech classification datasets
2. Running a training on the binary-labeled dataset:
```bash
    python3 ~/Predictor/predictor_BERT.py - --dataset_path ~/Predictor/data_classification/ --result_path ~/Predictor/results_classification/
```
3. Finding another hate-speech dataset
4. Cutting the phrases into prefixes and adding the relative weights:
```bash
    python3 ~/Predictor/utils/cut_phrases.py --input_csv ~/Predictor/data_prefixes/<dataset> --output_csv ~/Predictor/data_prefixes/<dataset_cut>
```
5. Hand-labeling the newly cut dataset with probability labels
6. Finetuning the previously trained model (step 2):
```bash
    python3 ~/Predictor/predictor_BERT_handcrafted.py - --dataset_path ~/Predictor/data_prefixes/ --result_path ~/Predictor/results_probabilities/ --load_model_dir ~/Predictor/results_classification/<model> --finetune
```
7. Finding another dataset for testing (or creating one), with prefixes and probability labels
8. Testing the trained and finetuned models:
```bash
    python3 ~/Predictor/predictor_BERT_handcrafted.py - --dataset_path ~/Predictor/data_test/ --result_path ~/Predictor/results_probabilities/ --load_model_dir ~/Predictor/results_classification/<model> --test
```
```bash
    python3 ~/Predictor/predictor_BERT_handcrafted.py - --dataset_path ~/Predictor/data_test/ --result_path ~/Predictor/results_probabilities/ --load_model_dir ~/Predictor/results_classification/<model_finetuned> --test
```
