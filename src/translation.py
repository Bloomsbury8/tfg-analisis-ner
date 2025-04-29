import os
import json
import logging
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

# Nota TFG: Se usan pipelines de Helsinki-NLP para back-translation es->en->es

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Librería Transformers no encontrada. Funcionalidad de traducción deshabilitada.")
    def pipeline(*args, **kwargs):
        raise ImportError("Se requiere la librería Transformers para los pipelines de traducción.")

from .data_models import Entry, ClinAISDataset, SectionAnnotation, BoundaryAnnotation
from .utils import split_entry, get_text_splits, getBoudaryAnnotationsForRange

# Inicialización diferida (lazy) de los pipelines
pipe_en_es = None
pipe_es_en = None

def get_translation_pipelines():
    """Inicializa y devuelve los pipelines de traducción."""
    global pipe_en_es, pipe_es_en
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("No se pueden obtener pipelines: Librería Transformers no instalada.")

    # Considerar añadir selección de dispositivo (e.g., CUDA si está disponible)
    device = -1 # CPU por defecto para evitar problemas de memoria en GPU pequeñas

    if pipe_en_es is None:
        logging.info("Inicializando pipeline en->es...")
        try:
            pipe_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=device)
        except Exception as e:
            logging.error(f"Fallo al cargar Helsinki-NLP/opus-mt-en-es: {e}")
            raise
    if pipe_es_en is None:
        logging.info("Inicializando pipeline es->en...")
        try:
            pipe_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=device)
        except Exception as e:
            logging.error(f"Fallo al cargar Helsinki-NLP/opus-mt-es-en: {e}")
            raise
    return pipe_es_en, pipe_en_es

def apply_translation_pipeline(text_list: List[str]) -> List[str]:
    """Aplica los pipelines es->en y en->es a una lista de textos."""
    if not text_list:
        return []
    pipe_es_en, pipe_en_es = get_translation_pipelines()
    logging.info(f"Traduciendo {len(text_list)} segmentos de texto (es->en->es)...")

    try:
        logging.debug("Ejecutando traducción es->en...")
        # Batching para eficiencia
        first_step_results = pipe_es_en(text_list, batch_size=16)
        english_texts = [res['translation_text'] for res in first_step_results]
    except Exception as e:
        logging.error(f"Error durante traducción es->en: {e}", exc_info=True)
        return ["" for _ in text_list] # Devolver placeholders

    try:
        logging.debug("Ejecutando traducción en->es...")
        second_step_results = pipe_en_es(english_texts, batch_size=16)
        spanish_texts = [res['translation_text'] for res in second_step_results]
        return spanish_texts
    except Exception as e:
        logging.error(f"Error durante traducción en->es: {e}", exc_info=True)
        return ["" for _ in text_list] # Devolver placeholders

def translate_entry(entry: Entry) -> Tuple[Dict[int, List[int]], List[str]]:
    """Divide secciones de una entrada si es necesario, las traduce y devuelve mapeo y textos.

    Returns:
        Tuple[Dict[int, List[int]], List[str]]:
            - Mapeo del índice de sección original a los índices de los textos (posiblemente divididos) en la lista devuelta.
            - Lista de textos traducidos (es->en->es).
    """
    if not TRANSFORMERS_AVAILABLE:
         logging.warning("Omitiendo translate_entry: Transformers no disponible.")
         return {}, []

    pipe_es_en, _ = get_translation_pipelines()
    tokenizer_max_len = getattr(pipe_es_en.tokenizer, 'model_max_length', 512)
    # Reducir límite para dejar margen a tokens especiales del pipeline
    max_accepted_length = int(0.7 * tokenizer_max_len)

    section2indeces: Dict[int, List[int]] = {}
    entry_sections_texts: List[str] = []
    next_index = 0

    logging.debug(f"Procesando entrada {entry.note_id} para preparación de traducción...")

    for i, section in enumerate(entry.section_annotation.gold):
        segment_text = str(section.segment) if section.segment is not None else ""
        if not segment_text.strip():
            logging.debug(f"Omitiendo segmento vacío en entrada {entry.note_id}, sección {i}")
            section2indeces[i] = []
            continue

        try:
            tokenized_section = pipe_es_en.tokenizer(segment_text, truncation=False, add_special_tokens=False)
            token_ids = tokenized_section['input_ids']

            if len(token_ids) > max_accepted_length:
                logging.debug(f"Segmento {i} en {entry.note_id} demasiado largo ({len(token_ids)} tokens > {max_accepted_length}), dividiendo...")
                bas_for_section = getBoudaryAnnotationsForRange(
                    entry.boundary_annotation.gold,
                    section.start_offset,
                    section.end_offset + 1 # Asumir inclusivo
                )

                if not bas_for_section:
                     logging.warning(f"No hay BoundaryAnnotations para sección larga {i} en {entry.note_id}. Usando texto original (puede fallar traducción).")
                     entry_sections_texts.append(segment_text)
                     section2indeces[i] = [next_index]
                     next_index += 1
                     continue

                # Dividir recursivamente usando la utilidad que chequea longitud de tokens
                rdy, splitted_bas = split_entry([bas_for_section], pipe_es_en.tokenizer, max_accepted_length)

                if not rdy:
                    logging.warning(f"La división de sección {i} en {entry.note_id} podría no haber resuelto completamente los límites de longitud.")

                text_splits_list = get_text_splits(entry, splitted_bas)
                if not text_splits_list:
                    logging.warning(f"Se obtuvieron splits de texto vacíos para sección {i} en {entry.note_id}. Usando original.")
                    entry_sections_texts.append(segment_text)
                    section2indeces[i] = [next_index]
                    next_index += 1
                else:
                    entry_sections_texts.extend(text_splits_list)
                    indices_for_this_section = list(range(next_index, next_index + len(text_splits_list)))
                    section2indeces[i] = indices_for_this_section
                    next_index += len(text_splits_list)
            else:
                # Longitud aceptable
                entry_sections_texts.append(segment_text)
                section2indeces[i] = [next_index]
                next_index += 1
        except Exception as e:
            logging.error(f"Error tokenizando/preparando sección {i} en entrada {entry.note_id}: {e}", exc_info=True)
            section2indeces[i] = [] # Marcar como fallido

    # Filtrar textos vacíos antes de traducir
    original_indices_map = {idx: text for idx, text in enumerate(entry_sections_texts) if text and not text.isspace()}
    valid_texts_to_translate = list(original_indices_map.values())

    if not valid_texts_to_translate:
         logging.warning(f"No se encontraron segmentos de texto válidos para traducir en entrada {entry.note_id}")
         return section2indeces, []

    translations = apply_translation_pipeline(valid_texts_to_translate)

    # Reconstruir lista final con placeholders para textos omitidos/vacíos
    final_translations = ["" for _ in entry_sections_texts]
    valid_translation_idx = 0
    for original_idx, original_text in enumerate(entry_sections_texts):
        if original_idx in original_indices_map:
            if valid_translation_idx < len(translations):
                final_translations[original_idx] = translations[valid_translation_idx]
                valid_translation_idx += 1
            else:
                logging.error(f"Falta resultado de traducción para índice {original_idx} en entrada {entry.note_id}. ¿Desfase de conteo?")

    if valid_translation_idx != len(translations):
         logging.error(f"Desfase de conteo ({valid_translation_idx} vs {len(translations)}) tras mapeo en entrada {entry.note_id}")

    return section2indeces, final_translations

def translate_dataset_and_save(dataset_path, translated_dataset_path):
   """Traduce un fichero de dataset entrada por entrada, guardando el progreso.

   Permite reanudar si el fichero de salida ya existe.
   """
   if not TRANSFORMERS_AVAILABLE:
        logging.error("No se puede traducir dataset: Librería Transformers no instalada.")
        return

   output_dir = os.path.dirname(translated_dataset_path)
   if output_dir and not os.path.exists(output_dir):
       logging.info(f"Creando directorio de salida para datos traducidos: {output_dir}")
       os.makedirs(output_dir, exist_ok=True)

   # Inicializar fichero traducido si no existe
   if not os.path.isfile(translated_dataset_path):
       logging.info(f"No existe {translated_dataset_path}; creando fichero nuevo.")
       try:
           with open(translated_dataset_path, 'w', encoding='utf-8') as f:
               f.write(ClinAISDataset(annotated_entries={}).toJson())
       except Exception as e:
           logging.error(f"Fallo al crear fichero traducido inicial {translated_dataset_path}: {e}")
           return
   else:
       logging.info(f"Encontrado fichero traducido existente {translated_dataset_path}. Se añadirán nuevas entradas.")

   # Cargar dataset original
   logging.info(f"Cargando dataset original desde {dataset_path}...")
   try:
       with open(dataset_path, 'r', encoding='utf-8') as f:
           ds = ClinAISDataset(**json.load(f))
   except FileNotFoundError:
       logging.error(f"Fichero de dataset original no encontrado: {dataset_path}")
       return
   except Exception as e:
       logging.error(f"Fallo al cargar dataset original {dataset_path}: {e}")
       return

   # Cargar dataset parcialmente traducido (si existe)
   logging.info(f"Cargando entradas previamente traducidas (si existen) desde {translated_dataset_path}...")
   try:
       with open(translated_dataset_path, 'r', encoding='utf-8') as f:
           translated_ds = ClinAISDataset(**json.load(f))
       logging.info(f"Cargadas {len(translated_ds.annotated_entries)} entradas previamente traducidas.")
   except Exception as e:
       logging.warning(f"Fallo al cargar dataset parcial {translated_dataset_path}. Empezando de nuevo. Error: {e}")
       translated_ds = ClinAISDataset(annotated_entries={})

   # Bucle de traducción
   logging.info("Iniciando proceso de traducción...")
   keys_list = list(ds.annotated_entries.keys())
   save_interval = 50 # Frecuencia de guardado
   entries_processed_this_run = 0
   entries_skipped = 0

   for idx, key in enumerate(tqdm(keys_list, desc="Traduciendo Entradas")):
       translated_key = key + "_T" # Clave para la entrada traducida
       if translated_key in translated_ds.annotated_entries:
           logging.debug(f"Omitiendo entrada {key}, versión traducida {translated_key} ya existe.")
           entries_skipped += 1
           continue

       entry = ds.annotated_entries[key]
       logging.debug(f"Traduciendo entrada {key}...")
       try:
           sec2idx, trans_texts = translate_entry(entry)

           if not trans_texts: # Si la traducción falló o no había texto válido
               logging.warning(f"No se generaron textos traducidos para la entrada {key}. Omitiendo guardado de esta entrada.")
               continue

           # Reconstruir la entrada traducida
           translated_entry = Entry(
               note_id=translated_key,
               note_text="", # Se reconstruirá más adelante
               section_annotation=SectionAnnotations(gold=[], prediction=[]), # Limpiar anotaciones originales
               boundary_annotation=BoundaryAnnotations(gold=[], prediction=[])
           )

           translated_sections_gold = []
           current_offset = 0
           for i_orig, section_orig in enumerate(entry.section_annotation.gold):
                if i_orig in sec2idx:
                    indices_in_trans = sec2idx[i_orig]
                    if not indices_in_trans: continue # Sección original omitida o vacía

                    translated_segment_parts = [trans_texts[j] for j in indices_in_trans if j < len(trans_texts)]
                    # Unir las partes traducidas (con espacio) y eliminar espacios extra al inicio/final
                    full_translated_segment = " ".join(translated_segment_parts).strip()

                    if not full_translated_segment:
                        # Si después de unir, el segmento está vacío, omitir.
                        logging.debug(f"Segmento traducido vacío para sección original {i_orig} en {key}. Omitiendo.")
                        continue

                    new_section = SectionAnnotation(
                        segment=full_translated_segment,
                        label=section_orig.label,
                        start_offset=current_offset,
                        end_offset=current_offset + len(full_translated_segment) -1 # Offset final es inclusivo
                    )
                    translated_sections_gold.append(new_section)
                    current_offset += len(full_translated_segment) + 1 # Añadir 1 por el espacio/separador asumido
                # else: Sección original no mapeada (probablemente vacía)

           translated_entry.section_annotation.gold = translated_sections_gold
           # Reconstruir note_text a partir de los segmentos traducidos unidos
           translated_entry.note_text = " ".join([s.segment for s in translated_sections_gold if s.segment]).strip()

           # Guardar la entrada traducida en el diccionario
           translated_ds.annotated_entries[translated_key] = translated_entry
           entries_processed_this_run += 1
           logging.debug(f"Entrada {translated_key} añadida al dataset traducido.")

       except Exception as e:
           logging.error(f"Error procesando/traduciendo la entrada {key}: {e}", exc_info=True)

       # Guardar progreso periódicamente
       if (idx + 1) % save_interval == 0 or idx == len(keys_list) - 1:
           logging.info(f"Guardando progreso... ({idx + 1}/{len(keys_list)} procesadas, {entries_processed_this_run} nuevas en esta ejecución)")
           try:
               with open(translated_dataset_path, 'w', encoding='utf-8') as f:
                    f.write(translated_ds.toJson())
           except Exception as e:
               logging.error(f"Fallo al guardar progreso en {translated_dataset_path}: {e}")

   logging.info(f"Proceso de traducción completado. Total entradas procesadas: {idx + 1}")
   logging.info(f"Entradas nuevas traducidas en esta ejecución: {entries_processed_this_run}")
   logging.info(f"Entradas omitidas (ya traducidas): {entries_skipped}")
   logging.info(f"Dataset traducido final guardado en: {translated_dataset_path}")


def create_augmented_dataset(train_set_path, translated_train_set_path, save_path):
   """Combina el dataset de entrenamiento original con el traducido para crear un dataset aumentado."""
   logging.info(f"Creando dataset aumentado en {save_path}...")

   output_dir = os.path.dirname(save_path)
   if output_dir and not os.path.exists(output_dir):
       logging.info(f"Creando directorio de salida para datos aumentados: {output_dir}")
       os.makedirs(output_dir, exist_ok=True)

   # Cargar dataset original
   logging.info(f"Cargando dataset original desde {train_set_path}...")
   try:
       with open(train_set_path, 'r', encoding='utf-8') as f:
           original_ds = ClinAISDataset(**json.load(f))
   except FileNotFoundError:
       logging.error(f"Fichero original no encontrado: {train_set_path}")
       return
   except Exception as e:
       logging.error(f"Fallo al cargar dataset original {train_set_path}: {e}")
       return

   # Cargar dataset traducido
   logging.info(f"Cargando dataset traducido desde {translated_train_set_path}...")
   try:
       with open(translated_train_set_path, 'r', encoding='utf-8') as f:
           translated_ds = ClinAISDataset(**json.load(f))
   except FileNotFoundError:
       logging.error(f"Fichero traducido no encontrado: {translated_train_set_path}")
       return
   except Exception as e:
       logging.error(f"Fallo al cargar dataset traducido {translated_train_set_path}: {e}")
       return

   # Combinar entradas
   augmented_entries = {}
   augmented_entries.update(original_ds.annotated_entries)
   augmented_entries.update(translated_ds.annotated_entries)

   augmented_ds = ClinAISDataset(annotated_entries=augmented_entries)
   logging.info(f"Dataset aumentado creado con {len(augmented_ds.annotated_entries)} entradas.")

   # Guardar dataset aumentado
   logging.info(f"Guardando dataset aumentado en {save_path}...")
   try:
       with open(save_path, 'w', encoding='utf-8') as f:
           f.write(augmented_ds.toJson())
       logging.info("Dataset aumentado guardado correctamente.")
   except Exception as e:
       logging.error(f"Fallo al guardar el dataset aumentado en {save_path}: {e}") 