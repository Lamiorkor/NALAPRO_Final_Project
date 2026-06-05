# NALAPRO Final Project

Individual final project for the Natural Language Processing Spring '26 class by **Naa Lamiorkor Boye**.

This repository contains the code, experiment outputs, and documentation for text classification experiments on the `20 Newsgroups` dataset from `sklearn.datasets.fetch_20newsgroups`.

## Project Overview

The project compares several NLP classification approaches:

1. A simple neural network with two linear layers and a ReLU activation using:
   - Word2Vec document embeddings
   - TF-IDF features
   - TF-IDF n-grams as an extra Task 1 experiment
2. BERT-base fine-tuning for supervised classification.
3. BERT masked-language-model adaptation followed by classification fine-tuning.
4. Llama-3 zero-shot and few-shot classification.
5. Bonus: Llama-3 QLoRA fine-tuning for classification.

The dataset is loaded directly through scikit-learn. The dataset itself is not included in this repository.

## Repository Structure

```text
nlp_tasks_1_and_2.ipynb
nlp_tasks_3_and_4.ipynb
nlp_bonus_llama3_qlora_task.ipynb
presentation.html
outputs/
   bert_confusion_matrix.png
   bert_experiment_results.csv
   bert_task2_results.csv
   bonus_llama3_comparison.csv
   bonus_llama3_comparison_f1.png
   bonus_llama3_qlora_confusion_matrix.png
   bonus_llama3_qlora_predictions.csv
   bonus_llama3_qlora_results.csv
   final_comparison_sorted_by_f1.csv
   final_comparison_sorted_by_f1.png
   task3_bert_mlm_results.csv
   task3_confusion_matrix.png
   task4_llama3_few_shot_confusion.png
   task4_llama3_few_shot_predictions.csv
   task4_llama3_zero_shot_confusion.png
   task4_llama3_zero_shot_full_context_confusion.png
   task4_llama3_zero_shot_full_context_predictions.csv
   task4_llama3_zero_shot_predictions.csv
   tasks_3_4_comparison_results.csv
   tasks_3_4_comparison_results_with_full_context.csv
```

## Notebooks

### `nlp_tasks_1_and_2.ipynb`

Covers Task 1 and Task 2.

Main contents:

- Data loading and preprocessing
- Word2Vec training across multiple epoch settings
- Word vector visualization
- Simple neural network classification using Word2Vec document vectors
- TF-IDF vectorization with the same simple neural network
- Extra Task 1 experiment using TF-IDF `(1, 2)`-grams
- BERT-base supervised fine-tuning
- BERT hyperparameter experiments
- W&B logging

### `nlp_tasks_3_and_4.ipynb`

Covers Task 3 and Task 4.

Main contents:

- BERT masked-language-model training on the 20 Newsgroups training text
- Classification fine-tuning using the MLM-adapted BERT checkpoint
- Llama-3 zero-shot classification
- Llama-3 few-shot classification
- Extra analysis: Llama-3 zero-shot classification with full document context, including headers, footers, and quotes
- Final comparison table sorted by weighted F1
- W&B logging

The main Llama-3 zero-shot and few-shot experiments use the cleaned test documents, matching the preprocessing used for the other models. The full-context Llama experiment is included only as an extra analysis because it uses additional document context.

### `nlp_bonus_llama3_qlora_task.ipynb`

Covers the optional bonus experiment.

Main contents:

- Llama-3 loading with 4-bit quantization
- QLoRA adapter fine-tuning using PEFT/TRL
- Prompt-completion classification format
- Evaluation on a held-out test subset
- W&B logging

For runtime reasons, the bonus experiment was run on a subset:

- Training samples: `2000`
- Validation samples: `200`
- Test samples: `500`
- Maximum document characters: `800`

## Experiment Tracking

Weights & Biases was used for experiment tracking.

W&B project:

```text
https://wandb.ai/naalamiorkor-boye-hochschule-luzern/nalapro-project
```

The notebooks log model configurations, training metrics, evaluation metrics, and selected result tables. Full model artifacts are not required for this submission and are not included in the repository.

## Main Results

The final comparison is saved in:

```text
outputs/final_comparison_sorted_by_f1.csv
```

The strongest main-model result was:

```text
BERT MLM-adapted then fine-tuned
Accuracy: 74.01%
Weighted F1: 0.7370
```

The supervised BERT baseline achieved:

```text
BERT base fine-tuned
Accuracy: 73.28%
Weighted F1: 0.7289
```

The best Task 1 neural-network result was:

```text
TF-IDF (1,2)-grams + no empty docs
Accuracy: 68.75%
Weighted F1: 0.6870
```

The bonus QLoRA experiment achieved:

```text
Llama-3 QLoRA fine-tuned
Training samples: 2000
Evaluation samples: 500
Accuracy: 75.00%
Weighted F1: 0.7434
```

Because the QLoRA result uses a smaller evaluation subset, it is not treated as directly equivalent to the full-test BERT comparisons.

## How to Run

The notebooks were developed and run in Google Colab with GPU acceleration. A GPU runtime is strongly recommended.

### 1. Open a Notebook in Google Colab

Upload or open one of the notebooks from the repository:

```text
nlp_tasks_1_and_2.ipynb
nlp_tasks_3_and_4.ipynb
nlp_bonus_llama3_qlora_task.ipynb
```

Then select:

```text
Runtime > Change runtime type > GPU
```

For BERT experiments, a T4/L4/A100 GPU is suitable. For Llama-3 and QLoRA experiments, an A100 GPU is preferred.

### 2. Set Up W&B

Create a Weights & Biases account and API key. In Colab, add the key under the secrets panel:

```text
WANDB_API_KEY
```

The notebooks use:

```python
wandb.login()
```

or load the key from Colab secrets if available.

### 3. Set Up Hugging Face for Llama-3

For the Llama-3 notebooks, create a Hugging Face account and access token. Add the token in Colab secrets:

```text
HF_TOKEN
```

You also need access to the gated Llama-3 model used in the notebooks:

```text
meta-llama/Meta-Llama-3-8B-Instruct
```

Request/accept model access on Hugging Face before running the Llama sections.

### 4. Install Dependencies

Each notebook includes its own install cell. The main packages used are:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
gensim
torch
transformers
datasets
accelerate
evaluate
wandb
huggingface_hub
bitsandbytes
peft
trl
```

`bitsandbytes`, `peft`, and `trl` are only needed for the Llama/QLoRA notebooks.

### 5. Run Order

Recommended order:

1. Run `nlp_tasks_1_and_2.ipynb`.
2. Run `nlp_tasks_3_and_4.ipynb`.
3. Optionally run `nlp_bonus_llama3_qlora_task.ipynb`.

The notebooks save CSV and PNG outputs into output folders, and the final selected outputs are included in `outputs/`.

## Notes on Reproducibility

- The dataset is downloaded through scikit-learn at runtime.
- Random seeds are set in the notebooks.
- GPU type, library versions, and Colab runtime behavior may cause small differences in training results.
- Llama-3 prompt-based experiments are especially sensitive to prompt format, parsing logic, and selected examples.

## Tools Used

- Google Colab with GPU runtime
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- NLTK
- Gensim Word2Vec
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face PEFT / TRL
- bitsandbytes
- Weights & Biases
- ChatGPT / AI assistance for debugging, explanation, code structuring, and result interpretation

## AI Use Statement

AI assistance was used to help debug errors, explain concepts, improve code structure, and support the interpretation of results. All code was reviewed, adapted, and understood before inclusion.

## Code Sources

Most code was written and adapted for this project. External libraries were used through their official APIs. The bonus QLoRA workflow was informed by the DataCamp tutorial on fine-tuning Llama 3.1 and adapted to the 20 Newsgroups classification task.

Reference:

```text
https://www.datacamp.com/tutorial/fine-tuning-llama-3-1
```
