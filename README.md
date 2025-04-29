# Segmentación de documentos clínicos en categorías útiles para tareas biomédicas

This project provides a comprehensive pipeline for fine-tuning transformer models (from the Hugging Face Hub) for the task of **Clinical Section Identification** in Spanish medical documents. It leverages the ClinAIS dataset format and focuses on achieving robust segmentation performance, evaluated primarily using the official **Weighted B2** boundary similarity metric.

The pipeline includes data processing specific to the ClinAIS format, handling of long sequences, optional data augmentation via back-translation, training using the Hugging Face `Trainer`, and detailed evaluation including both token-level and boundary-based metrics.

## Features

*   **Fine-tuning:** Trains transformer models for token classification on clinical section identification tasks.
*   **Data Handling:** Processes datasets in the ClinAIS JSON format (`ClinAISDataset`, `Entry`, `BoundaryAnnotation`, `SectionAnnotation` Pydantic models).
*   **Sequence Splitting:** Implements robust logic (`WordListSplitter`, `EntrySplitter`, recursive token checks) to handle clinical notes longer than the model's maximum input length by splitting them intelligently based on sentence boundaries and token limits.
*   **Data Augmentation (Optional):**
    *   **Back-Translation:** Includes an optional step to augment the training data by translating it to English and back to Spanish using Helsinki-NLP models (`opus-mt-es-en`, `opus-mt-en-es`). This helps improve model robustness.
    *   **Augmented Dataset Creation:** Combines the original training data with the back-translated data.
*   **Evaluation:**
    *   **Token-level:** Calculates standard token classification metrics (Precision, Recall, F1, Accuracy) using `seqeval` during training evaluation.
    *   **Boundary-based (Weighted B2):** Integrates with an external `metricas.py` script (specifically the `score_predictions` function) to calculate the official Weighted B2 boundary score during training callbacks and final evaluation.
*   **Training Callbacks:**
    *   **Early Stopping:** Stops training if the validation metric (`eval_f1`) doesn't improve for a set number of epochs.
    *   **Boundary B2 Evaluation:** Custom callback to compute and log the Weighted B2 score on the validation set after each evaluation epoch.
*   **Modular Structure:** Code is organized into logical modules within the `src/` directory for better maintainability.
*   **Command-Line Interface:** Uses `argparse` for easy configuration of file paths, model names, training hyperparameters, and optional steps.
*   **Hugging Face Integration:** Downloads models and tokenizers from the Hub, handles login (CLI or token).

## Project Structure

```
.
├── main.py                 # Main script entry point, handles arguments and orchestration
├── requirements.txt        # Python package dependencies
├── README.md               # This file
├── metricas.py             
└── src/                    # Source code modules
    ├── __init__.py         # Makes src a Python package
    ├── data_models.py      # Pydantic models for dataset structure (ClinAISDataset, Entry, etc.)
    ├── splitters.py        # WordListSplitter and EntrySplitter classes
    ├── utils.py            # Utility functions (e.g., recursive split_entry)
    ├── translation.py      # Data augmentation via back-translation
    ├── data_preparation.py # Dataset creation, tokenization, and alignment logic
    ├── postprocessing.py   # Prediction post-processing logic (merging sections, etc.)
    ├── prediction.py       # Generating predictions on entries and datasets
    ├── training.py         # Trainer building, metrics calculation, BoundaryB2Callback
    └── evaluation.py       # Final model evaluation logic (test_model function)
```

**Note:** The `metricas.py` script containing the `score_predictions` function is **required** for the boundary-based evaluation (Weighted B2 score) used in callbacks and final testing. This script is assumed to be external and **must be placed** in a location accessible by the script (either the root directory or a path specified via `--metricas_path`). It was originally part of the [Sec-Identification-in-Spanish-Clinical-Notes](https://github.com/Iker0610/Sec-Identification-in-Spanish-Clinical-Notes).

## Setup

1.  **Clone/Download:** Get the project code onto your local machine.
2.  **Python Version:** Python 3.10 or higher is recommended (developed with 3.11).
3.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *This includes `transformers`, `torch`, `datasets`, `evaluate`, `seqeval`, `pydantic`, etc.*
5.  **Place `metricas.py`:** Obtain the `metricas.py` file containing the `score_predictions(prediction_file: Path, output_result_file: Path)` function and place it:
    *   In the project root directory (alongside `main.py`).
    *   *OR* in a custom directory and specify its path using the `--metricas_path` argument when running `main.py`.
6.  **Hugging Face Login (Optional but Recommended):**
    *   For downloading models (especially private ones) and potentially pushing results later, log in to the Hugging Face Hub:
        ```bash
        huggingface-cli login
        ```
    *   Alternatively, you can provide an access token directly using the `--hf_token` argument when running the script.

## Data Format

The script expects input data (`--original_train_path`, `--val_data_path`, `--test_data_path`) in a JSON format that can be parsed into the `ClinAISDataset` Pydantic model defined in `src/data_models.py`. This generally involves a dictionary of `Entry` objects, each containing `note_id`, `note_text`, and annotations (`section_annotation`, `boundary_annotation`).

## Usage

The script is executed via `main.py` with various command-line arguments.

```bash
python main.py --original_train_path <path> --val_data_path <path> --output_dir <path> [OPTIONS]
```

### Required Arguments:

*   `--original_train_path` (str): Path to the original training JSON file (e.g., `data/clinais.train.json`).
*   `--val_data_path` (str): Path to the validation JSON file (e.g., `data/clinais.dev.json`).
*   `--output_dir` (str): Directory where the trained model, tokenizer, predictions, metrics, and intermediate data (like translated/augmented files) will be saved.

### Optional Arguments:

**File Paths:**

*   `--test_data_path` (str): Path to the test JSON file for final evaluation. If not provided, the validation set (`--val_data_path`) is used for the final boundary-based evaluation. (Default: `None`)
*   `--metricas_path` (str): Directory containing the `metricas.py` script. (Default: `.`)
*   `--translated_train_path` (str): Path to save/load the translated training data. (Default: `[output_dir]/data/[basename].translated.json`)
*   `--augmented_train_path` (str): Path to save/load the augmented training data. (Default: `[output_dir]/data/[basename].augmented.json`)

**Model & Training:**

*   `--base_model` (str): Hugging Face model identifier for the base model. (Default: `PlanTL-GOB-ES/bsc-bio-ehr-es`)
*   `--num_train_epochs` (int): Number of training epochs. (Default: `6`)
*   `--learning_rate` (float): Learning rate. (Default: `3e-5`)
*   `--train_batch_size` (int): Batch size per device during training. (Default: `8`)
*   `--eval_batch_size` (int): Batch size for evaluation. (Default: `16`)
*   `--gradient_accumulation` (int): Gradient accumulation steps. (Default: `2`)

**Augmentation:**

*   `--skip_translation` (bool flag): Skip the data translation step entirely. (Default: `False`)
*   `--skip_augmentation` (bool flag): Skip creating the augmented dataset (uses original train path, or translated path if translation was run but augmentation skipped). (Default: `False`)

**Hugging Face:**

*   `--hf_token` (str): Optional Hugging Face API token for login. (Default: `None`) some models need to be adquired.

### Examples:

1.  **Basic Training (No Augmentation):**
    ```bash
    python main.py \
        --original_train_path data/clinais.train.json \
        --val_data_path data/clinais.dev.json \
        --output_dir results/basic_run \
        --skip_translation \
        --skip_augmentation
    ```

2.  **Training with Translation and Augmentation:**
    ```bash
    python main.py \
        --original_train_path data/clinais.train.json \
        --val_data_path data/clinais.dev.json \
        --output_dir results/augmented_run
        # --skip_translation and --skip_augmentation are False by default
    ```
    *(This will create `results/augmented_run/data/clinais.train.translated.json` and `results/augmented_run/data/clinais.train.augmented.json`)*

3.  **Using Pre-translated Data for Augmentation:**
    ```bash
    python main.py \
        --original_train_path data/clinais.train.json \
        --val_data_path data/clinais.dev.json \
        --output_dir results/pretranslated_aug_run \
        --skip_translation \
        --translated_train_path path/to/your/pretranslated.train.json # Specify existing translated file
        # --skip_augmentation is False by default
    ```

4.  **Specifying Custom Paths and Test Set:**
    ```bash
    python main.py \
        --original_train_path ~/datasets/train.json \
        --val_data_path ~/datasets/dev.json \
        --test_data_path ~/datasets/test.json \
        --output_dir ~/experiments/run1 \
        --metricas_path /path/to/evaluation_scripts/ \
        --skip_translation \
        --skip_augmentation
    ```

5.  **Freezing Base Model and Fewer Epochs:**
    ```bash
    python main.py \
        --original_train_path data/clinais.train.json \
        --val_data_path data/clinais.dev.json \
        --output_dir results/freeze_run \
        --num_train_epochs 3 \
        --do_freeze \
        --skip_translation \
        --skip_augmentation
    ```

6.  **Using a Different Base Model and Providing HF Token:**
    ```bash
    python main.py \
        --original_train_path data/clinais.train.json \
        --val_data_path data/clinais.dev.json \
        --output_dir results/mbert_run \
        --base_model bert-base-multilingual-cased \
        --hf_token YOUR_HUGGINGFACE_TOKEN \
        --skip_translation \
        --skip_augmentation
    ```

## Workflow

The `main.py` script orchestrates the following steps:

1.  **Setup:** Parses arguments, configures logging, checks for `metricas.py`, handles HF login.
2.  **Data Augmentation (Optional):**
    *   If `--skip_translation` is not set, `src.translation.translate_dataset_and_save` is called.
    *   If `--skip_augmentation` is not set (and translation was successful or skipped but a translated file exists), `src.translation.create_augmented_dataset` is called.
    *   The path to the training data to be used (`train_data_to_use`) is determined based on these steps.
3.  **Build Trainer:** `src.training.build_trainer` is called, which involves:
    *   Loading the tokenizer (`AutoTokenizer`).
    *   Executing the data preparation pipeline (`src.data_preparation.execute_data_preparation_pipeline`):
        *   Loading raw data (`ClinAISDataset`).
        *   Creating Hugging Face `Dataset` objects.
        *   Tokenizing and aligning labels, handling sequence length limits via splitting (`tokenize_dataset_dict`).
    *   Loading the model (`AutoModelForTokenClassification`).
    *   Creating the `DataCollator`.
    *   Configuring `TrainingArguments`.
    *   Setting up callbacks (`EarlyStoppingCallback`, `BoundaryB2Callback`).
    *   Initializing the `Trainer`.
4.  **Train:** `trainer.train()` is executed. Checkpoints and logs (e.g., TensorBoard) are saved to the `--output_dir`.
5.  **Save Model:** The final best model, tokenizer, and trainer state are saved to `--output_dir`.
6.  **Evaluate (Token-Level):** `trainer.evaluate()` is called to compute seqeval metrics on the validation set. Results are logged and saved.
7.  **Evaluate (Boundary-Based):** `src.evaluation.test_model` is called:
    *   Loads the fine-tuned model from `--output_dir`.
    *   Creates a prediction pipeline.
    *   Generates predictions on the specified test set (or validation set if no test set provided) using `src.prediction.create_predictions_file`.
    *   Calls the external `score_predictions` function (imported from `metricas.py`) to calculate Weighted B2 and other boundary metrics.
    *   Logs the final Weighted B2 score.

## Output Files

The `--output_dir` will contain:

*   **Checkpoints:** Subdirectories like `checkpoint-XXXX` containing model weights, tokenizer files, trainer state, etc., saved during training.
*   **Final Model:** The best model and tokenizer files saved directly in the output directory after training completes.
*   **Logs:** TensorBoard logs (usually in a `runs/` subdirectory) if `report_to` includes `"tensorboard"`.
*   **Metrics:** JSON files containing training and evaluation metrics (`train_results.json`, `all_results.json`, etc.).
*   **Callback Evaluations (if run):** Subdirectories like `callback_eval_epoch_X.XX/` containing `predictions.json` and `evaluation_results.json` generated by the `BoundaryB2Callback`.
*   **Final Evaluation Files:**
    *   `final_predictions.json`: Predictions generated by `test_model` on the final test/validation set.
    *   `final_evaluation_results.json`: Evaluation results (including Weighted B2) generated by the `score_predictions` function called within `test_model`.
*   **Data Subdirectory:**
    *   `data/`: Contains intermediate data files if augmentation is used.
    *   `[basename].translated.json`: Output of the translation step.
    *   `[basename].augmented.json`: Output of the augmentation step.

## Customization

*   **Hyperparameters:** Adjust training parameters like learning rate, batch size, epochs, etc., using the command-line arguments.
*   **Base Model:** Change the base model using `--base_model`.
*   **Code Modification:** For more fundamental changes (e.g., different splitting logic, data models, evaluation metrics), modify the code within the relevant `src/` modules. 
