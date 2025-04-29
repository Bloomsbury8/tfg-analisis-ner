import os
import json
import logging
from pathlib import Path
from typing import Callable, Any # Para type hint de score_predictions

# Use try-except for optional dependencies
try:
    import transformers
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Librería Transformers no encontrada. Funcionalidad de evaluación deshabilitada.")
    # Definiciones dummy
    def pipeline(*args, **kwargs):
        raise ImportError("Librería Transformers requerida para pipeline.")
    def AutoModelForTokenClassification(*args, **kwargs):
         raise ImportError("Librería Transformers requerida para AutoModelForTokenClassification.")
    def AutoTokenizer(*args, **kwargs):
         raise ImportError("Librería Transformers requerida para AutoTokenizer.")

# Relative imports
from .prediction import create_predictions_file

# Define type hint for the external scoring function
ScorePredictionsFunc = Callable[[Path, Path], Any]

def test_model(
    finetuned_model_path: str,
    test_dataset_path: str,
    save_predictions_path: str,
    save_evaluated_path: str,
    score_predictions_func: ScorePredictionsFunc,
    aggregation_strategy: str = "simple"
):
   """Carga un modelo fine-tuned, ejecuta predicciones y evalúa usando una función externa."""
   if not TRANSFORMERS_AVAILABLE:
        logging.error("No se puede ejecutar test_model: Librería Transformers no instalada.")
        return

   logging.info(f"--- Iniciando Evaluación Final del Modelo ---")
   logging.info(f" Path Modelo: {finetuned_model_path}")
   logging.info(f" Dataset Test: {test_dataset_path}")
   logging.info(f" Path Predicciones: {save_predictions_path}")
   logging.info(f" Path Resultados Eval: {save_evaluated_path}")

   # Validar entradas
   if not os.path.isdir(finetuned_model_path):
       logging.error(f"Directorio del modelo no encontrado: {finetuned_model_path}")
       return
   if not os.path.isfile(test_dataset_path):
       logging.error(f"Fichero de dataset de test no encontrado: {test_dataset_path}")
       return

   # 1. Cargar modelo y tokenizer
   try:
       logging.info("Cargando modelo y tokenizer fine-tuned...")
       model = AutoModelForTokenClassification.from_pretrained(finetuned_model_path)
       tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
       logging.info("Modelo y tokenizer cargados.")
   except Exception as e:
       logging.error(f"Fallo al cargar modelo/tokenizer desde {finetuned_model_path}: {e}", exc_info=True)
       return

   # 2. Crear pipeline de predicción
   try:
       # Determinar dispositivo (GPU si disponible, si no CPU)
       device = 0 if transformers.is_torch_available() and transformers.torch.cuda.is_available() else -1
       pipeline_model = pipeline(
           "token-classification",
           model=model,
           tokenizer=tokenizer,
           aggregation_strategy=aggregation_strategy,
           device=device
       )
       logging.info(f"Pipeline de predicción creado en device {device} con aggregation='{aggregation_strategy}'.")
   except Exception as e:
       logging.error(f"Fallo al crear pipeline de predicción: {e}", exc_info=True)
       return

   # 3. Generar fichero de predicciones
   try:
       logging.info(f"Generando predicciones...")
       pred_dir = os.path.dirname(save_predictions_path)
       if pred_dir: os.makedirs(pred_dir, exist_ok=True)
       # Llamar a la función del módulo prediction
       create_predictions_file(test_dataset_path, save_predictions_path, pipeline_model, tokenizer)
       logging.info(f"Predicciones guardadas en {save_predictions_path}")
   except FileNotFoundError as e:
       logging.error(f"Error durante generación de predicciones: ¿Fichero de entrada no encontrado? {e}")
       return
   except Exception as e:
       logging.error(f"Fallo durante generación de predicciones: {e}", exc_info=True)
       return

   # 4. Evaluar usando la función externa proporcionada
   if not os.path.exists(save_predictions_path):
        logging.error(f"Fichero de predicciones no encontrado en {save_predictions_path}. No se puede evaluar.")
        return

   try:
       logging.info(f"Evaluando predicciones con la función de scoring proporcionada...")
       eval_dir = os.path.dirname(save_evaluated_path)
       if eval_dir: os.makedirs(eval_dir, exist_ok=True)

       # Llamar a la función externa (score_predictions)
       score_predictions_func(prediction_file=Path(save_predictions_path), output_result_file=Path(save_evaluated_path))
       logging.info(f"Resultados de evaluación guardados en {save_evaluated_path}")
   except FileNotFoundError:
        logging.error(f"Función de scoring no encontró el fichero de predicciones {save_predictions_path}.")
        return
   except ImportError:
        logging.error("ImportError durante ejecución de función de scoring. ¿Está 'metricas' accesible?")
        return
   except Exception as e:
       logging.error(f"Error durante ejecución de script de evaluación externo: {e}", exc_info=True)
       return

   # 5. Mostrar puntuación Weighted B2 si está disponible en el fichero de resultados
   try:
       if os.path.exists(save_evaluated_path):
            with open(save_evaluated_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            # Buscar la clave específica usada por metricas.py
            weighted_b2 = results.get("Weighted B2") # Ajustar clave si es necesario
            if weighted_b2 is not None:
                 try:
                      b2_float = float(weighted_b2)
                      logging.info(f"--- Puntuación Final Weighted B2: {b2_float:.4f} ---")
                 except (ValueError, TypeError):
                      logging.warning(f"Score Weighted B2 encontrado ('{weighted_b2}'), pero no se pudo convertir a float.")
            else:
                 logging.warning(f"Score 'Weighted B2' no encontrado en fichero de resultados: {save_evaluated_path}. Keys: {list(results.keys())}")
       else:
            logging.warning(f"Fichero de resultados de evaluación no creado o no encontrado: {save_evaluated_path}")
   except json.JSONDecodeError as e:
       logging.error(f"Error leyendo JSON de resultados de evaluación {save_evaluated_path}: {e}")
   except Exception as e:
       logging.error(f"Error procesando fichero de resultados de evaluación: {e}", exc_info=True)

   logging.info("--- Evaluación Final del Modelo Finalizada ---") 