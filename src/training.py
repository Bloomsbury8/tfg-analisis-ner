import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from pathlib import Path

# Use try-except for optional dependencies
try:
    import evaluate
    import seqeval
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    logging.warning("Librerías evaluate y/o seqeval no encontradas. Métricas seqeval deshabilitadas.")

try:
    import transformers
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        TrainingArguments,
        DataCollatorForTokenClassification,
        Trainer,
        pipeline,
        EarlyStoppingCallback,
        TrainerCallback,
        PreTrainedModel,
        PreTrainedTokenizerBase
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Librería transformers no encontrada. Funcionalidad de entrenamiento deshabilitada.")
    # Explicitly define needed types if transformers is missing, mostly for type hinting
    class TrainerCallback: pass
    class TrainingArguments: pass
    class Trainer: pass
    class PreTrainedModel: pass
    class PreTrainedTokenizerBase: pass


# Relative imports for project modules
from .data_models import ClinicalSections
from .data_preparation import execute_data_preparation_pipeline
from .prediction import create_predictions_file # Needed for callback

# Type hint for score_predictions function (assuming its signature)
ScorePredictionsFunc = Callable[[Path, Path], Any] # Input: pred_path, output_path -> results

def get_seqeval_metrics(id2label: Dict[int, str]) -> Callable[[Any], Dict[str, float]]:
   """Devuelve una función compute_metrics para seqeval usando la librería evaluate."""
   if not EVALUATE_AVAILABLE:
        logging.error("No se pueden calcular métricas seqeval: evaluate/seqeval no encontradas.")
        def dummy_compute_metrics(p):
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        return dummy_compute_metrics

   try:
        seqeval_metric = evaluate.load("seqeval")
   except Exception as e:
        logging.error(f"Fallo al cargar métrica seqeval desde evaluate: {e}")
        def dummy_compute_metrics(p):
             return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        return dummy_compute_metrics

   def compute_metrics_fn(p):
       predictions, labels = p
       predictions = np.argmax(predictions, axis=2)

       # Convertir IDs a strings, ignorando -100
       true_predictions = [
           [id2label.get(p, "UNK") for (p, l) in zip(prediction, label) if l != -100]
           for prediction, label in zip(predictions, labels)
       ]
       true_labels = [
           [id2label.get(l, "UNK") for (_, l) in zip(prediction, label) if l != -100]
           for prediction, label in zip(predictions, labels)
       ]

       try:
            # Evitar error si alguna lista está vacía tras filtrar -100
            if not any(true_predictions) or not any(true_labels):
                 logging.warning("Predicciones/labels vacías tras filtrar -100. Devolviendo métricas cero.")
                 return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

            results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
            return {
               "precision": results.get("overall_precision", 0.0),
               "recall": results.get("overall_recall", 0.0),
               "f1": results.get("overall_f1", 0.0),
               "accuracy": results.get("overall_accuracy", 0.0),
            }
       except Exception as e:
            logging.error(f"Error calculando métricas seqeval: {e}", exc_info=True)
            # Log para debug
            try:
                 logging.error(f"Ejemplo forma Preds: {len(true_predictions[0]) if true_predictions else 'N/A'}, Labels: {len(true_labels[0]) if true_labels else 'N/A'}")
                 logging.error(f"Ejemplo contenido Preds: {true_predictions[0][:10] if true_predictions and true_predictions[0] else 'N/A'}")
                 logging.error(f"Ejemplo contenido Labels: {true_labels[0][:10] if true_labels and true_labels[0] else 'N/A'}")
            except: pass
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

   return compute_metrics_fn

def create_label_id_dictionaries(section_types: List[str]) -> Tuple[Dict[str, int], Dict[int, str], int]:
   """Crea los diccionarios label2id y id2label.

   Asegura que la lista `section_types` incluye todas las etiquetas deseadas (incl. 'O' si se usa).
   """
   label2id = {label: i for i, label in enumerate(section_types)}
   id2label = {i: label for i, label in enumerate(section_types)}
   num_labels = len(section_types)
   logging.info(f"Creados mapeos con {num_labels} etiquetas: {list(label2id.keys())}")
   if num_labels == 0: logging.warning("¡Mapeo de etiquetas creado con cero etiquetas!")
   return label2id, id2label, num_labels

# Custom Callback for Boundary B2 Evaluation during training
class BoundaryB2Callback(TrainerCallback):
   """Callback para calcular la métrica Weighted B2 en cada evaluación."""
   def __init__(self, \
                val_data_path: str, \
                output_dir: str, \
                score_predictions_func: ScorePredictionsFunc, \
                aggregation_strategy: str = "simple"):
       if not TRANSFORMERS_AVAILABLE:
            raise ImportError("BoundaryB2Callback requiere la librería transformers.")
       self.val_data_path = val_data_path
       self.output_dir = output_dir # Para guardar ficheros temporales
       self.score_predictions_func = score_predictions_func
       self.aggregation_strategy = aggregation_strategy
       self.trainer: Trainer | None = None

   def set_trainer(self, trainer: Trainer):
       self.trainer = trainer

   def on_evaluate(self, args: TrainingArguments, state, control, metrics=None, **kwargs):
       """Se ejecuta tras la fase de evaluación para calcular Weighted B2."""
       if self.trainer is None or not state.is_world_process_zero:
            logging.debug("BoundaryB2Callback: Omitiendo cálculo B2 (no es proceso principal o trainer no asignado).")
            return

       if not os.path.exists(self.val_data_path):
            logging.error(f"BoundaryB2Callback: Fichero de validación {self.val_data_path} no encontrado. No se puede calcular B2.")
            return

       current_epoch = state.epoch if state.epoch is not None else "unknown"
       logging.info(f"BoundaryB2Callback: Ejecutando evaluación boundary para epoch {current_epoch}...")

       model: PreTrainedModel | None = self.trainer.model
       tokenizer: PreTrainedTokenizerBase | None = self.trainer.tokenizer
       if model is None or tokenizer is None:
           logging.warning("BoundaryB2Callback: Modelo o tokenizer del trainer no disponibles.")
           return

       # Crear pipeline con el modelo/tokenizer actual
       try:
           # Need transformers.pipeline if available
           if not TRANSFORMERS_AVAILABLE:
                logging.error("BoundaryB2Callback: transformers.pipeline no disponible.")
                return
           device = args.device
           pipe_model = pipeline(
               "token-classification",
               model=model,
               tokenizer=tokenizer,
               aggregation_strategy=self.aggregation_strategy,
               device=device
           )
       except Exception as e:
           logging.error(f"BoundaryB2Callback: Fallo al crear pipeline de predicción: {e}", exc_info=True)
           return

       # Definir paths temporales en subdirectorio
       eval_epoch_dir = os.path.join(self.output_dir, f"callback_eval_epoch_{current_epoch:.2f}" if isinstance(current_epoch, float) else f"callback_eval_epoch_{current_epoch}")
       try:
            os.makedirs(eval_epoch_dir, exist_ok=True)
            predictions_file = os.path.join(eval_epoch_dir, "predictions.json")
            evaluated_file = os.path.join(eval_epoch_dir, "evaluation_results.json")
       except OSError as e:
           logging.error(f"BoundaryB2Callback: Fallo al crear directorio {eval_epoch_dir}: {e}")
           return

       try:
           logging.info(f"BoundaryB2Callback: Generando predicciones para {self.val_data_path}...")
           create_predictions_file(self.val_data_path, predictions_file, pipe_model, tokenizer)

           logging.info(f"BoundaryB2Callback: Puntuando predicciones con función externa...")
           self.score_predictions_func(prediction_file=Path(predictions_file), output_result_file=Path(evaluated_file))

           if os.path.exists(evaluated_file):
                with open(evaluated_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                weighted_b2 = results.get("Weighted B2")

                if weighted_b2 is not None:
                     try:
                          weighted_b2_float = float(weighted_b2)
                          logging.info(f"Epoch {current_epoch:.2f}: Weighted B2 => {weighted_b2_float:.4f}")
                          if metrics is not None:
                              metrics["weighted_b2"] = weighted_b2_float # Añadir a métricas del Trainer
                     except (ValueError, TypeError):
                          logging.error(f"Epoch {current_epoch:.2f}: Valor de Weighted B2 '{weighted_b2}' no es float válido.")
                else:
                     logging.warning(f"Epoch {current_epoch:.2f}: Score 'Weighted B2' no encontrado en {evaluated_file}. Keys: {results.keys()}")
           else:
                logging.error(f"BoundaryB2Callback: Fichero de resultados de evaluación no creado: {evaluated_file}")

       except FileNotFoundError as e:
           logging.error(f"BoundaryB2Callback: Fichero no encontrado durante cálculo B2: {e}")
       except ImportError as e:
           logging.error(f"BoundaryB2Callback: ImportError durante puntuación (inesperado): {e}")
       except Exception as e:
           logging.error(f"BoundaryB2Callback: Error durante cálculo Weighted B2: {e}", exc_info=True)

       # Nota TFG: No limpiar ficheros temporales para poder inspeccionarlos.

def build_trainer(base_model: str, train_data_path: str, val_data_path: str,
                 out_dir: str, score_predictions_func: ScorePredictionsFunc,
                 do_freeze: bool = False, training_args_dict: dict = None) -> Trainer:
   """Construye y configura el Trainer de Hugging Face."""
   if not TRANSFORMERS_AVAILABLE:
        raise ImportError("build_trainer requiere la librería transformers.")

   logging.info(f"Construyendo trainer con modelo base: {base_model}")

   # 1. Crear Mapeo de Etiquetas (basado en Enum ClinicalSections)
   all_labels = ClinicalSections.list()
   label2id, id2label, num_labels = create_label_id_dictionaries(all_labels)

   # 2. Cargar Tokenizer
   logging.info(f"Cargando tokenizer para {base_model}...")
   try:
        # Need transformers.AutoTokenizer if available
        if not TRANSFORMERS_AVAILABLE:
             raise ImportError("AutoTokenizer no disponible.")
        tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)
        # Nota TFG: add_prefix_space=True puede ser importante para algunos modelos (ej: RoBERTa)
   except Exception as e:
       logging.error(f"Fallo al cargar tokenizer {base_model}: {e}", exc_info=True)
       raise

   # 3. Preparar Datasets (Carga, Tokenización, Alineación, Splitting)
   logging.info("Ejecutando pipeline de preparación de datos...")
   try:
        tokenized_datasets = execute_data_preparation_pipeline(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            tokenizer=tokenizer,
            label2id=label2id
        )
   except Exception as e:
       logging.error(f"Fallo en el pipeline de preparación de datos: {e}", exc_info=True)
       raise

   # 4. Cargar Modelo
   logging.info(f"Cargando modelo {base_model} para Token Classification...")
   try:
        # Need transformers.AutoModelForTokenClassification if available
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("AutoModelForTokenClassification no disponible.")
        model = AutoModelForTokenClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
   except Exception as e:
       logging.error(f"Fallo al cargar modelo {base_model}: {e}", exc_info=True)
       raise

   # 5. Congelar capas base si se especifica (freeze)
   if do_freeze:
       logging.info("Congelando capas del modelo base (entrenando solo el clasificador)...")
       try:
           # Asumiendo estructura estándar de HF (ej: modelo.bert, modelo.roberta)
           base_model_attr = model.base_model_prefix
           if hasattr(model, base_model_attr):
               for param in getattr(model, base_model_attr).parameters():
                   param.requires_grad = False
               logging.info(f"Capas de '{base_model_attr}' congeladas.")
           else:
                logging.warning(f"No se pudo encontrar el atributo del modelo base '{base_model_attr}' para congelar. Continuando sin congelar.")
       except Exception as e:
            logging.warning(f"Error intentando congelar capas: {e}. Continuando sin congelar.")

   # 6. Definir Argumentos de Entrenamiento
   logging.info("Configurando argumentos de entrenamiento...")
   default_training_args = {
        "output_dir": out_dir,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "num_train_epochs": 6,
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1", # Usar F1 de seqeval para early stopping
        "greater_is_better": True,
        "push_to_hub": False,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "report_to": ["tensorboard"] # Opcional: "wandb"
        # Añadir más argumentos por defecto si se necesitan
   }
   # Sobrescribir con argumentos del diccionario proporcionado
   if training_args_dict:
       logging.info(f"Sobrescribiendo argumentos por defecto con: {training_args_dict}")
       default_training_args.update(training_args_dict)

   # Need transformers.TrainingArguments if available
   if not TRANSFORMERS_AVAILABLE:
        raise ImportError("TrainingArguments no disponible.")
   training_args = TrainingArguments(**default_training_args)

   # 7. Data Collator
   # Need transformers.DataCollatorForTokenClassification if available
   if not TRANSFORMERS_AVAILABLE:
        raise ImportError("DataCollatorForTokenClassification no disponible.")
   data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

   # 8. Función Compute Metrics (seqeval)
   compute_metrics = get_seqeval_metrics(id2label)

   # 9. Callbacks
   # Need transformers.EarlyStoppingCallback if available
   if not TRANSFORMERS_AVAILABLE:
        raise ImportError("EarlyStoppingCallback no disponible.")
   early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
   boundary_eval_callback = BoundaryB2Callback(
       val_data_path=val_data_path,
       output_dir=out_dir,
       score_predictions_func=score_predictions_func
   )

   # 10. Inicializar Trainer
   logging.info("Inicializando Trainer...")
   # Need transformers.Trainer if available
   if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Trainer no disponible.")
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_datasets["train"],
       eval_dataset=tokenized_datasets["validation"],
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
       callbacks=[early_stopping, boundary_eval_callback] # Añadir callbacks
   )

   # Adjuntar trainer al callback para que tenga acceso
   boundary_eval_callback.set_trainer(trainer)

   logging.info("Trainer construido exitosamente.")
   return trainer 