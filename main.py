import os
import sys
import logging
import argparse
import json

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a transformer model for clinical section identification.")

    # Paths
    parser.add_argument("--original_train_path", type=str, required=True, help="Path to the original training JSON file (e.g., clinais.train.json)")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation JSON file (e.g., clinais.dev.json)")
    parser.add_argument("--test_data_path", type=str, default=None, help="Optional: Path to the test JSON file for final evaluation. Uses validation set if not provided.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--metricas_path", type=str, default=".", help="Directory containing 'metricas.py'.")

    # Model & Training
    parser.add_argument("--base_model", type=str, default="PlanTL-GOB-ES/bsc-bio-ehr-es", help="Base Hugging Face model.")
    parser.add_argument("--do_freeze", action='store_true', help="Freeze base model layers.")
    parser.add_argument("--num_train_epochs", type=int, default=6, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps.")

    # Augmentation
    parser.add_argument("--skip_translation", action='store_true', help="Skip data translation.")
    parser.add_argument("--skip_augmentation", action='store_true', help="Skip augmented dataset creation.")
    parser.add_argument("--translated_train_path", type=str, default=None, help="Path for translated data. Default: '[output_dir]/data/[basename].translated.json'")
    parser.add_argument("--augmented_train_path", type=str, default=None, help="Path for augmented data. Default: '[output_dir]/data/[basename].augmented.json'")

    # HF Login
    parser.add_argument("--hf_token", type=str, default=None, help="Optional: Hugging Face API token.")

    return parser.parse_args()

# --- Main Orchestration Logic ---
def main():
    args = parse_arguments()

    # Importar funcion de metricas externa
    sys.path.insert(0, args.metricas_path)
    try:
        from src.metricas import score_predictions
        logging.info(f"Successfully imported score_predictions from {args.metricas_path}")
    except ImportError:
        logging.error(f"Error: Could not import 'score_predictions' from 'metricas.py' in '{args.metricas_path}'.")
        logging.error("Ensure 'metricas.py' exists and contains 'score_predictions'.")
        sys.exit(1)
    except Exception as e:
         logging.error(f"Unexpected error importing score_predictions: {e}")
         sys.exit(1)

    # Importar modulos del proyecto
    try:
        from src.translation import translate_dataset_and_save, create_augmented_dataset
        from src.training import build_trainer
        from src.evaluation import test_model
    except ModuleNotFoundError as e:
         logging.error(f"Error importing source modules: {e}. Check 'src' structure.")
         sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during module imports: {e}")
        sys.exit(1)

    # Login en Hugging Face (opcional)
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            logging.info("Logged in to Hugging Face Hub using token.")
        except ImportError:
            logging.warning("huggingface_hub not found. Cannot login with token.")
        except Exception as e:
            logging.error(f"Failed to login to Hugging Face Hub with token: {e}")
    else:
        try:
            from huggingface_hub import whoami
            whoami()
            logging.info("Already logged in to Hugging Face Hub (CLI detected).")
        except ImportError:
            logging.warning("huggingface_hub not found. Cannot check CLI login status.")
        except Exception:
             logging.warning("Not logged in to Hugging Face Hub. Private models might fail.")
             logging.warning("Use 'huggingface-cli login' or provide --hf_token.")

    # Definir paths de datos
    os.makedirs(args.output_dir, exist_ok=True)
    data_subdir = os.path.join(args.output_dir, "data")
    os.makedirs(data_subdir, exist_ok=True)

    original_basename = os.path.basename(args.original_train_path)
    default_translated_path = os.path.join(data_subdir, original_basename.replace(".json", ".translated.json"))
    default_augmented_path = os.path.join(data_subdir, original_basename.replace(".json", ".augmented.json"))

    translated_train_path = args.translated_train_path or default_translated_path
    augmented_train_path = args.augmented_train_path or default_augmented_path

    train_data_to_use = args.original_train_path # Path inicial

    # Pasos 1 & 2: Augmentation (Opcional)
    # Logica compleja para determinar qué fichero de entrenamiento usar basado en flags
    if not args.skip_translation:
        logging.info("--- Starting Data Translation --- ")
        try:
            translate_dataset_and_save(args.original_train_path, translated_train_path)
            logging.info("--- Finished Data Translation --- ")
            translation_successful = True
        except Exception as e:
            logging.error(f"Data translation failed: {e}", exc_info=True)
            translation_successful = False
            logging.warning("Proceeding without translated data due to error.")

        if translation_successful and not args.skip_augmentation:
            logging.info("--- Starting Data Augmentation --- ")
            try:
                create_augmented_dataset(args.original_train_path, translated_train_path, augmented_train_path)
                train_data_to_use = augmented_train_path
                logging.info(f"--- Finished Data Augmentation (Using: {train_data_to_use}) --- ")
            except Exception as e:
                 logging.error(f"Data augmentation failed: {e}", exc_info=True)
                 logging.warning("Proceeding with original training data due to augmentation error.")
                 train_data_to_use = args.original_train_path
        elif args.skip_augmentation:
             logging.info("Skipping augmentation step.")
             if translation_successful:
                 logging.info(f"Using translated data directly: {translated_train_path}")
                 train_data_to_use = translated_train_path
             else:
                 logging.info("Using original training data.")
                 train_data_to_use = args.original_train_path

    elif not args.skip_augmentation:
         # Translation skipped, but augmentation requested with existing translated file
         logging.info("Translation skipped, attempting augmentation with provided translated file.")
         logging.info(f"Expecting pre-translated file at: {translated_train_path}")
         if os.path.exists(translated_train_path):
             try:
                 create_augmented_dataset(args.original_train_path, translated_train_path, augmented_train_path)
                 train_data_to_use = augmented_train_path
                 logging.info(f"--- Finished Data Augmentation (Using: {train_data_to_use}) --- ")
             except Exception as e:
                 logging.error(f"Data augmentation failed: {e}", exc_info=True)
                 logging.warning("Proceeding with original training data due to augmentation error.")
                 train_data_to_use = args.original_train_path
         else:
             logging.warning(f"Provided translated file '{translated_train_path}' not found. Cannot augment.")
             logging.warning("Proceeding with original training data.")
             train_data_to_use = args.original_train_path
    else:
        logging.info("Translation and augmentation skipped. Using original training data.")
        train_data_to_use = args.original_train_path

    logging.info(f"Final training data path determined: {train_data_to_use}")

    # Paso 3: Construir y Entrenar
    logging.info("--- Starting Model Training --- ")
    try:
        training_args_override = {
           'num_train_epochs': args.num_train_epochs,
           'learning_rate': args.learning_rate,
           'per_device_train_batch_size': args.train_batch_size,
           'per_device_eval_batch_size': args.eval_batch_size,
           'gradient_accumulation_steps': args.gradient_accumulation,
           'output_dir': args.output_dir,
           # Otros argumentos de TrainingArguments pueden ir aqui si se añaden a parse_arguments
        }

        trainer = build_trainer(
           base_model=args.base_model,
           train_data_path=train_data_to_use,
           val_data_path=args.val_data_path,
           out_dir=args.output_dir,
           score_predictions_func=score_predictions, # Pasar funcion importada
           do_freeze=args.do_freeze,
           training_args_dict=training_args_override
        )

        logging.info("Starting trainer.train()...")
        train_result = trainer.train()
        logging.info("Training finished.")

        trainer.save_model(args.output_dir) # Guarda modelo y tokenizer
        trainer.save_state()
        logging.info(f"Final model, tokenizer, and trainer state saved to {args.output_dir}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    except Exception as e:
        logging.error(f"Build Trainer or Training failed: {e}", exc_info=True)
        sys.exit(1)

    # Paso 4: Evaluar metricas token-level (seqeval) en validacion
    logging.info("--- Evaluating final model (token-level seqeval) on validation set --- ")
    try:
        final_metrics = trainer.evaluate() # Usa eval_dataset del trainer
        logging.info(f"Final token-level eval metrics: {final_metrics}")
        trainer.log_metrics("eval", final_metrics)
        trainer.save_metrics("eval", final_metrics)
    except Exception as e:
        logging.error(f"Final token-level evaluation failed: {e}", exc_info=True)
        # Considerar si continuar con la evaluacion boundary

    # Paso 5: Evaluar metricas boundary (Weighted B2) en test/validacion
    logging.info("--- Evaluating final model (boundary-based B2) --- ")
    test_set_path_for_boundary = args.test_data_path if args.test_data_path else args.val_data_path
    logging.info(f"Using dataset for final boundary evaluation: {test_set_path_for_boundary}")

    final_predictions_json = os.path.join(args.output_dir, "final_predictions.json")
    final_evaluated_json = os.path.join(args.output_dir, "final_evaluation_results.json")

    try:
        test_model(
            finetuned_model_path=args.output_dir, # Directorio del modelo guardado
            test_dataset_path=test_set_path_for_boundary,
            save_predictions_path=final_predictions_json,
            save_evaluated_path=final_evaluated_json,
            score_predictions_func=score_predictions, # Pasar funcion importada
        )
    except Exception as e:
        logging.error(f"Final boundary-based evaluation (test_model) failed: {e}", exc_info=True)

    logging.info("--- Script finished --- ")

if __name__ == "__main__":
    main() 