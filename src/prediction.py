import os
import json
import logging
from typing import List
from enum import Enum
from tqdm import tqdm

# Use relative imports for modules within the src package
from .data_models import (
    Entry, ClinAISDataset, Prediction, PredictionSection,
    BoundaryAnnotation, SectionAnnotation, ClinicalSections
)
from .utils import split_entry, get_text_splits # Assuming split_entry handles token length checks
from .postprocessing import PredictionPostProcessor

# Assume pipeline and tokenizer type hints require:
# from transformers import Pipeline, PreTrainedTokenizerBase

def process_entry(entry: Entry, model_pipe, tokenizer) -> None:
    """Procesa una única entrada: divide, predice, post-procesa y actualiza anotaciones.

    Modifica el objeto 'entry' directamente (in-place).
    """
    logging.debug(f"Procesando entrada: {entry.note_id}")

    # 1. Dividir entrada basada en boundaries si excede longitud máxima
    if not entry.boundary_annotation or not entry.boundary_annotation.gold:
        logging.warning(f"Entrada {entry.note_id} sin boundary annotations gold. No se puede procesar.")
        entry.section_annotation.prediction = []
        entry.boundary_annotation.prediction = []
        return

    # Iniciar con lista completa de boundaries gold
    ba_list_to_split = [entry.boundary_annotation.gold]
    try:
        # split_entry maneja división recursiva por longitud de tokens
        rdy, split_ba_lists = split_entry(ba_list_to_split, tokenizer)
        if not rdy:
             logging.warning(f"La división de {entry.note_id} podría no haber resuelto problemas de longitud.")
        if not split_ba_lists:
            logging.error(f"La división de {entry.note_id} resultó en lista vacía. No se puede procesar.")
            entry.section_annotation.prediction = []
            entry.boundary_annotation.prediction = []
            return
    except Exception as e:
        logging.error(f"Error durante split_entry para {entry.note_id}: {e}", exc_info=True)
        entry.section_annotation.prediction = []
        entry.boundary_annotation.prediction = []
        return

    # 2. Obtener segmentos de texto correspondientes a los splits
    try:
        final_text_splits = get_text_splits(entry, split_ba_lists)
        if not final_text_splits:
             logging.warning(f"No se generaron splits de texto para {entry.note_id}. Omitiendo predicción.")
             entry.section_annotation.prediction = []
             entry.boundary_annotation.prediction = []
             return
    except Exception as e:
        logging.error(f"Error obteniendo text splits para {entry.note_id}: {e}", exc_info=True)
        entry.section_annotation.prediction = []
        entry.boundary_annotation.prediction = []
        return

    # 3. Ejecutar predicciones sobre los segmentos de texto
    all_raw_predictions = Prediction()
    logging.debug(f"Ejecutando predicciones en {len(final_text_splits)} splits para entrada {entry.note_id}...")

    # Calcular offset absoluto para cada predicción parcial
    current_split_index = 0
    for i, txt in enumerate(final_text_splits):
        if not txt or txt.isspace():
            logging.debug(f"Omitiendo split de texto vacío {i} para entrada {entry.note_id}")
            continue

        # Encontrar el offset absoluto de inicio de este split
        absolute_start_offset = -1
        if i < len(split_ba_lists) and split_ba_lists[i]:
            try:
                absolute_start_offset = split_ba_lists[i][0].start_offset
            except (AttributeError, IndexError):
                 logging.error(f"No se pudo obtener start_offset del primer boundary del split {i} en {entry.note_id}.")
                 continue # Saltar este split si no se puede obtener offset
        else:
            logging.error(f"Falta información de boundary para split {i} en {entry.note_id}. No se puede calcular offset.")
            continue

        if absolute_start_offset == -1:
             logging.error(f"Offset absoluto no calculado para split {i} en {entry.note_id}. Saltando predicción.")
             continue

        try:
            partial_output = model_pipe(txt)
            for p_dict in partial_output:
                 try:
                     p_start_relative = int(p_dict['start'])
                     p_end_relative = int(p_dict['end'])
                     p_start_absolute = absolute_start_offset + p_start_relative
                     # End es exclusivo en pipeline HF, mantenerlo así para consistencia interna temporal
                     p_end_absolute = absolute_start_offset + p_end_relative

                     label_str = p_dict['entity_group']
                     label_enum = ClinicalSections(label_str)

                     pred_sec = PredictionSection(
                         entity_group=label_enum,
                         score=float(p_dict.get('score', 0.0)),
                         word=p_dict['word'],
                         start=p_start_absolute,
                         end=p_end_absolute # Almacenar end exclusivo temporalmente
                     )
                     all_raw_predictions.sections.append(pred_sec)
                 except ValueError:
                      logging.warning(f"Entity group desconocido '{p_dict.get('entity_group')}' predicho en split {i}, entrada {entry.note_id}. Omitiendo.")
                 except KeyError as ke:
                      logging.warning(f"Clave {ke} faltante en salida de predicción: {p_dict}. Omitiendo.")
                 except Exception as e_conv:
                      logging.error(f"Error convirtiendo dict {p_dict} a Pydantic: {e_conv}")

        except Exception as e:
            logging.error(f"Error prediciendo en split {i} para entrada {entry.note_id}: {e}", exc_info=True)

    all_raw_predictions.sections.sort(key=lambda s: s.start)

    # 4. Post-procesar las predicciones combinadas
    logging.debug(f"Post-procesando {len(all_raw_predictions.sections)} predicciones raw para {entry.note_id}...")
    try:
        ppp = PredictionPostProcessor(all_raw_predictions, verbose=False)
        processed_predictions = ppp.do_all()
    except Exception as e:
        logging.error(f"Error durante post-procesado para {entry.note_id}: {e}", exc_info=True)
        processed_predictions = all_raw_predictions # Usar raw como fallback

    # 5. Actualizar entry.section_annotation.prediction
    entry.section_annotation.prediction = []
    for s in processed_predictions.sections:
        try:
            label_enum = s.entity_group if isinstance(s.entity_group, Enum) else ClinicalSections(s.entity_group)
            # Ajustar end offset de exclusivo a inclusivo para SectionAnnotation
            final_end_offset = s.end -1 if s.end > s.start else s.start

            entry.section_annotation.prediction.append(SectionAnnotation(
                segment=s.word,
                label=label_enum,
                start_offset=s.start,
                end_offset=final_end_offset # Guardar end inclusivo
            ))
        except Exception as e:
            logging.error(f"Error creando SectionAnnotation predicha para {entry.note_id}: {e}. Data: {s}")

    # 6. Actualizar entry.boundary_annotation.prediction
    try:
        # Crear copia limpia de boundaries gold para modificar
        b_pred = [BoundaryAnnotation(**gold_ba.dict()) for gold_ba in entry.boundary_annotation.gold]
        for ba in b_pred: ba.boundary = None # Limpiar etiquetas previas
    except Exception as e:
        logging.error(f"Error creando boundaries predichos iniciales para {entry.note_id}: {e}")
        b_pred = []

    boundaries_labeled = set()
    for sec in processed_predictions.sections:
        first_boundary_in_section_found = False
        sec_start = sec.start
        # Usar end exclusivo para el chequeo de rango [start, end)
        sec_end = sec.end

        for idx, ba in enumerate(b_pred):
             # Chequear si inicio del boundary cae en [sec_start, sec_end)
             if idx not in boundaries_labeled and ba.start_offset >= sec_start and ba.start_offset < sec_end:
                  if not first_boundary_in_section_found:
                       try:
                           label_enum = sec.entity_group if isinstance(sec.entity_group, Enum) else ClinicalSections(sec.entity_group)
                           ba.boundary = label_enum
                           boundaries_labeled.add(idx)
                           first_boundary_in_section_found = True
                           break # Etiquetar solo el primer boundary por sección
                       except Exception as e:
                            logging.error(f"Error asignando etiqueta boundary para {entry.note_id}: {e}. Sec: {sec}, Bound: {ba}")
             # Optimización: if ba.start_offset >= sec_end: break (si están ordenados)

    entry.boundary_annotation.prediction = b_pred
    logging.debug(f"Procesamiento finalizado para entrada: {entry.note_id}")

def create_predictions_file(dataset_path: str, save_predicted_path: str, model_pipeline, tokenizer) -> None:
   """Carga un dataset, genera predicciones para cada entrada y guarda el dataset actualizado."""
   logging.info(f"Cargando dataset para predicción desde: {dataset_path}")
   try:
       with open(dataset_path, 'r', encoding='utf-8') as f:
           ds = ClinAISDataset(**json.load(f))
       logging.info(f"Cargadas {len(ds.annotated_entries)} entradas.")
   except FileNotFoundError:
       logging.error(f"Fichero dataset no encontrado en {dataset_path}. No se pueden crear predicciones.")
       return
   except json.JSONDecodeError as e:
       logging.error(f"Error decodificando JSON desde {dataset_path}: {e}")
       return
   except Exception as e:
       logging.error(f"Error cargando dataset {dataset_path}: {e}")
       return

   logging.info("Generando predicciones para cada entrada...")
   entry_items = list(ds.annotated_entries.items())
   for key, entry in tqdm(entry_items, desc="Prediciendo Entradas"):
       try:
           process_entry(entry, model_pipeline, tokenizer)
       except Exception as e:
           logging.error(f"Error fatal procesando entrada {key}: {e}", exc_info=True)
           # Marcar entrada como fallida limpiando predicciones
           entry.section_annotation.prediction = []
           entry.boundary_annotation.prediction = []

   # Asegurar que directorio de salida existe
   output_dir = os.path.dirname(save_predicted_path)
   if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creando directorio para predicciones: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

   logging.info(f"Guardando predicciones en: {save_predicted_path}")
   try:
       with open(save_predicted_path, 'w', encoding='utf-8') as f:
           # Usar el método toJson de ClinAISDataset que maneja Pydantic
           f.write(ds.toJson())
       logging.info("Predicciones guardadas correctamente.")
   except Exception as e:
       logging.error(f"Fallo al guardar predicciones en {save_predicted_path}: {e}", exc_info=True)
 