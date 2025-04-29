import logging
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from datasets import Dataset, DatasetDict

from .data_models import ClinAISDataset, BoundaryAnnotations
from .splitters import WordListSplitter

# from transformers import PreTrainedTokenizerBase

def get_labelled_span_list(boundary_annotations: BoundaryAnnotations) -> List[tuple[str, str | None]]:
    """Extrae spans y etiquetas de BoundaryAnnotations.gold.

    Propaga la etiqueta del token de boundary a los tokens siguientes dentro de la misma sección.
    Devuelve [(span, label_string_or_None), ...].
    """
    spans_with_labels = []
    current_label_enum = None
    if not boundary_annotations or not boundary_annotations.gold:
        logging.warning("BoundaryAnnotations vacías o inválidas en get_labelled_span_list")
        return []

    for ba in boundary_annotations.gold:
        if ba.boundary is not None:
            current_label_enum = ba.boundary

        label_value = current_label_enum.value if current_label_enum else None
        span_text = str(ba.span) if ba.span is not None else ""
        spans_with_labels.append((span_text, label_value))

    return spans_with_labels

def create_dataset_object(data_set: ClinAISDataset, label2id: dict) -> Dataset:
   """Crea un objeto Dataset de Hugging Face a partir de ClinAISDataset."""
   all_spans = []
   all_labels = []
   skipped_entries = 0
   processed_entries = 0

   if not data_set or not data_set.annotated_entries:
       logging.warning("ClinAISDataset vacío o sin entradas.")
       return Dataset.from_dict({'spans': [], 'labels': []})

   logging.info(f"Procesando {len(data_set.annotated_entries)} entradas para creación de Dataset...")

   for entry_id, entry in tqdm(data_set.annotated_entries.items(), desc="Creando Dataset Object"):
        try:
            labeled_spans = get_labelled_span_list(entry.boundary_annotation)
            if not labeled_spans:
                logging.warning(f"No se encontraron spans etiquetados para entrada {entry_id}. Omitiendo.")
                skipped_entries += 1
                continue
        except Exception as e:
            logging.error(f"Error obteniendo spans etiquetados para entrada {entry_id}: {e}", exc_info=True)
            skipped_entries += 1
            continue

        entry_spans = []
        entry_label_ids = []
        possible = True
        for span, label_str in labeled_spans:
            span_str = str(span) if span is not None else ""
            entry_spans.append(span_str)

            label_id = -1
            if label_str is None:
                if "O" in label2id:
                    label_id = label2id["O"]
                elif None in label2id:
                    label_id = label2id[None]
                else:
                    logging.error(f"Etiqueta 'None' encontrada en {entry_id}, pero no hay mapeo (ej: 'O') en label2id: {list(label2id.keys())}. Omitiendo entrada.")
                    possible = False
                    break
            elif label_str not in label2id:
                logging.error(f"Etiqueta '{label_str}' en {entry_id} no encontrada en label2id: {list(label2id.keys())}. Omitiendo entrada.")
                possible = False
                break
            else:
                label_id = label2id[label_str]

            entry_label_ids.append(label_id)

        if possible:
            if len(entry_spans) != len(entry_label_ids):
                 logging.error(f"Error interno: Desfase entre spans ({len(entry_spans)}) y IDs ({len(entry_label_ids)}) para {entry_id}. Omitiendo.")
                 skipped_entries += 1
            elif not entry_spans:
                 logging.warning(f"Entrada {entry_id} resultó sin spans tras procesar. Omitiendo.")
                 skipped_entries += 1
            else:
                 all_spans.append(entry_spans)
                 all_labels.append(entry_label_ids)
                 processed_entries += 1
        else:
            skipped_entries += 1

   logging.info(f"Creación de Dataset finalizada. Procesadas: {processed_entries}, Omitidas: {skipped_entries}")

   if not all_spans:
        logging.error("No se procesaron entradas válidas. No se puede crear el Dataset.")
        return Dataset.from_dict({'spans': [], 'labels': []})

   if len(all_spans) != len(all_labels):
       logging.error(f"Error fatal: Desfase entre listas finales de spans ({len(all_spans)}) y labels ({len(all_labels)}). No se puede crear Dataset.")
       return Dataset.from_dict({'spans': [], 'labels': []})

   try:
        return Dataset.from_dict({'spans': all_spans, 'labels': all_labels})
   except Exception as e:
        logging.error(f"Fallo al crear el objeto Dataset: {e}", exc_info=True)
        return Dataset.from_dict({'spans': [], 'labels': []})

def get_reshaped_list(target_list, pattern_list):
   """Reestructura una lista plana 'target_list' según la estructura anidada de 'pattern_list'."""
   result = []
   current_index = 0
   for sublist in pattern_list:
        if not hasattr(sublist, '__len__'):
             logging.error(f"Sublista del patrón no es iterable: {sublist}. No se puede reestructurar.")
             return []
        sublist_length = len(sublist)

        if current_index + sublist_length > len(target_list):
             logging.error(f"No se puede reestructurar: patrón requiere {sublist_length} elementos desde índice {current_index}, pero solo quedan {len(target_list) - current_index}.")
             return []

        result.append(target_list[current_index : current_index + sublist_length])
        current_index += sublist_length

   if current_index != len(target_list):
        logging.warning(f"Reestructuración finalizada, pero {len(target_list) - current_index} elementos de la lista original no se usaron.")

   return result

def split_entry_for_tokenizer(entry_data: dict, word_splitter: WordListSplitter):
   """Divide 'spans' y 'labels' correspondientes usando WordListSplitter.

   Entrada: {'spans': [w1, w2,...], 'labels': [id1, id2,...]}
   Salida: {'spans': [[split1_w1,...], [split2_w1,...]], 'labels': [[split1_id1,...], [split2_id1,...]]}
   """
   if 'spans' not in entry_data or 'labels' not in entry_data:
        logging.error("Faltan 'spans' o 'labels' en entry_data para dividir.")
        return {'spans': [], 'labels': []}
   if len(entry_data['spans']) != len(entry_data['labels']):
        logging.error(f"Longitudes desfasadas en split_entry_for_tokenizer: {len(entry_data['spans'])} spans, {len(entry_data['labels'])} labels.")
        return {'spans': [], 'labels': []}

   split_spans_list = word_splitter.split(entry_data['spans'])
   if not split_spans_list:
        logging.warning("WordListSplitter devolvió lista vacía.")
        return {'spans': [], 'labels': []}

   split_labels_list = get_reshaped_list(entry_data['labels'], split_spans_list)

   if len(split_spans_list) != len(split_labels_list):
       logging.error("Desfase de longitud tras reestructurar etiquetas. Falló la división.")
       logging.error(f"Longitudes Originales: Spans={len(entry_data['spans'])}, Labels={len(entry_data['labels'])}")
       logging.error(f"Estructura Spans Divididos: {[len(s) for s in split_spans_list]}")
       logging.error(f"Conteo Labels Reestructurados: {len(split_labels_list)}")
       return {'spans': [], 'labels': []}

   for i, (spans_sublist, labels_sublist) in enumerate(zip(split_spans_list, split_labels_list)):
       if len(spans_sublist) != len(labels_sublist):
           logging.error(f"Desfase interno en sublista {i}: {len(spans_sublist)} spans vs {len(labels_sublist)} labels.")
           return {'spans': [], 'labels': []}

   return {'spans': split_spans_list, 'labels': split_labels_list}

def tokenize_and_align_labels(batch, tokenizer, label_all_tokens=False):
    """Tokeniza texto (ya dividido en palabras) y alinea etiquetas a tokens.

    Entrada batch: {'spans': [[w1, w2], [w3]], 'labels': [[l1, l2], [l3]]}
    Salida batch: {'input_ids': ..., 'attention_mask': ..., 'labels': ... (alineadas)}
    """
    if not all(isinstance(s, list) for s in batch['spans']):
        logging.error("batch['spans'] no es una lista de listas. Formato esperado: List[List[str]]")
        return {'input_ids': [], 'attention_mask': [], 'labels': [], 'original_spans': [], 'original_labels': []}

    tokenized_inputs = tokenizer(
        batch['spans'],
        truncation=False,
        is_split_into_words=True,
        padding=False
    )

    aligned_labels_batch = []
    input_ids_batch = []
    attention_mask_batch = []
    original_spans_batch = []
    original_labels_batch = []

    for i, labels_for_entry in enumerate(batch['labels']):
        original_spans = batch['spans'][i]
        try:
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            current_input_ids = tokenized_inputs.input_ids[i]
            current_attention_mask = tokenized_inputs.attention_mask[i]
        except IndexError:
            logging.error(f"IndexError accediendo a resultados tokenizados para índice {i}. ¿Tamaño de batch inconsistente?")
            continue
        except Exception as e:
             logging.error(f"Error obteniendo word_ids/tokens para índice {i}: {e}")
             continue

        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(labels_for_entry):
                    label_ids.append(labels_for_entry[word_idx])
                else:
                    logging.error(f"Índice de palabra {word_idx} fuera de rango para etiquetas (len {len(labels_for_entry)}) en índice {i}. Usando -100.")
                    label_ids.append(-100)
            else:
                if label_all_tokens:
                     if word_idx < len(labels_for_entry):
                        label_ids.append(labels_for_entry[word_idx])
                     else:
                        label_ids.append(-100)
                else:
                     label_ids.append(-100)

            previous_word_idx = word_idx

        # Nota TFG: Control de longitud post-tokenización
        if len(current_input_ids) > tokenizer.model_max_length:
             logging.warning(f"Secuencia tokenizada excede max_length ({len(current_input_ids)} > {tokenizer.model_max_length}) para índice {i} DESPUÉS de alinear. Considerar truncamiento.")
             continue

        aligned_labels_batch.append(label_ids)
        input_ids_batch.append(current_input_ids)
        attention_mask_batch.append(current_attention_mask)
        original_spans_batch.append(original_spans)
        original_labels_batch.append(labels_for_entry)

    final_batch = {
        'input_ids': input_ids_batch,
        'attention_mask': attention_mask_batch,
        'labels': aligned_labels_batch,
    }

    return final_batch

def check_and_split_if_needed(batch, tokenizer, word_splitter):
    """Comprueba si alguna entrada en el batch excede la longitud máxima antes de tokenizar.

    Si excede, usa WordListSplitter para dividirla en sub-entradas.
    Devuelve un nuevo batch donde las entradas largas se reemplazan por sus splits.
    Formato entrada/salida: {'spans': List[List[str]], 'labels': List[List[int]]}
    """
    new_spans_batch = []
    new_labels_batch = []
    max_len = tokenizer.model_max_length

    for i in range(len(batch['spans'])):
        current_spans = batch['spans'][i]
        current_labels = batch['labels'][i]

        if len(current_spans) <= word_splitter.max_size:
            new_spans_batch.append(current_spans)
            new_labels_batch.append(current_labels)
            continue

        logging.debug(f"Entrada {i} tiene {len(current_spans)} spans (potencialmente larga), aplicando split_entry_for_tokenizer...")
        split_result = split_entry_for_tokenizer({'spans': current_spans, 'labels': current_labels}, word_splitter)

        if split_result['spans']:
            new_spans_batch.extend(split_result['spans'])
            new_labels_batch.extend(split_result['labels'])
            logging.debug(f" -> Entrada {i} dividida en {len(split_result['spans'])} sub-entradas.")
        else:
            logging.warning(f"Fallo al dividir entrada {i} (original {len(current_spans)} spans). Omitiendo entrada.")

    new_batch = {
        'spans': new_spans_batch,
        'labels': new_labels_batch
    }
    return new_batch

def tokenize_dataset_dict(ds_dict: DatasetDict, tokenizer, max_retries=3) -> DatasetDict:
    """Tokeniza y alinea etiquetas para un DatasetDict, manejando splits si es necesario."""

    initial_max_words = int(tokenizer.model_max_length * 0.8)
    min_words = max(1, initial_max_words // 4)
    word_splitter = WordListSplitter(max_size=initial_max_words, min_size=min_words)

    logging.info("Aplicando chequeo/división inicial basado en número de palabras...")
    ds_dict_split_checked = ds_dict.map(
        lambda batch: check_and_split_if_needed(batch, tokenizer, word_splitter),
        batched=True,
        desc="Checking/Splitting Entries by Word Count"
    )

    logging.info("Tokenizando y alineando etiquetas...")
    tokenized_aligned_ds = ds_dict_split_checked.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, label_all_tokens=False),
        batched=True,
        remove_columns=ds_dict_split_checked["train"].column_names,
        desc="Tokenizing and Aligning Labels"
    )

    logging.info("Validando longitudes finales de secuencias tokenizadas...")
    max_len = tokenizer.model_max_length
    for split_name in tokenized_aligned_ds.keys():
        num_removed = 0
        original_count = len(tokenized_aligned_ds[split_name])
        def filter_long_sequences(example):
            return len(example['input_ids']) <= max_len

        tokenized_aligned_ds[split_name] = tokenized_aligned_ds[split_name].filter(
            filter_long_sequences,
            desc=f"Filtering long sequences in {split_name}"
        )
        num_removed = original_count - len(tokenized_aligned_ds[split_name])
        if num_removed > 0:
             logging.warning(f"Se eliminaron {num_removed} secuencias de '{split_name}' por exceder max_length ({max_len}) tras todo el proceso.")

    logging.info("Tokenización y alineación completadas.")
    return tokenized_aligned_ds

def execute_data_preparation_pipeline(train_data_path: str, val_data_path: str, tokenizer, label2id: dict) -> DatasetDict:
    """Carga datos, crea Datasets, tokeniza y alinea etiquetas.

    Args:
        train_data_path: Path al fichero JSON de entrenamiento (formato ClinAISDataset).
        val_data_path: Path al fichero JSON de validación (formato ClinAISDataset).
        tokenizer: Tokenizador Hugging Face pre-entrenado.
        label2id: Diccionario de mapeo de etiquetas string a IDs numéricos.

    Returns:
        DatasetDict: Contiene los datasets 'train' y 'validation' procesados y listos para el Trainer.
                   Las columnas son 'input_ids', 'attention_mask', 'labels'.
    """
    logging.info("--- Iniciando Pipeline de Preparación de Datos ---")

    logging.info(f"Cargando datos de entrenamiento desde: {train_data_path}")
    try:
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_clinais_ds = ClinAISDataset(**json.load(f))
    except FileNotFoundError:
        logging.error(f"Fichero de entrenamiento no encontrado: {train_data_path}")
        raise
    except Exception as e:
        logging.error(f"Error cargando datos de entrenamiento: {e}", exc_info=True)
        raise

    logging.info(f"Cargando datos de validación desde: {val_data_path}")
    try:
        with open(val_data_path, 'r', encoding='utf-8') as f:
            val_clinais_ds = ClinAISDataset(**json.load(f))
    except FileNotFoundError:
        logging.error(f"Fichero de validación no encontrado: {val_data_path}")
        raise
    except Exception as e:
        logging.error(f"Error cargando datos de validación: {e}", exc_info=True)
        raise

    logging.info("Creando objetos Dataset de Hugging Face...")
    try:
        train_dataset = create_dataset_object(train_clinais_ds, label2id)
        val_dataset = create_dataset_object(val_clinais_ds, label2id)
    except Exception as e:
        logging.error(f"Error creando objetos Dataset: {e}", exc_info=True)
        raise

    if len(train_dataset) == 0 or len(val_dataset) == 0:
         logging.error("Uno o ambos datasets están vacíos tras la creación inicial. No se puede continuar.")
         raise ValueError("Error: Datasets vacíos tras la conversión inicial.")

    ds_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    logging.info(f"DatasetDict creado. Train: {len(ds_dict['train'])} entradas, Validation: {len(ds_dict['validation'])} entradas.")
    logging.info(f"Columnas iniciales: {ds_dict['train'].column_names}")

    logging.info("Iniciando tokenización y alineación de etiquetas...")
    try:
        tokenized_ds_dict = tokenize_dataset_dict(ds_dict, tokenizer)
    except Exception as e:
        logging.error(f"Error durante la tokenización/alineación: {e}", exc_info=True)
        raise

    logging.info(f"Dataset final listo. Train: {len(tokenized_ds_dict['train'])}, Validation: {len(tokenized_ds_dict['validation'])}.")
    logging.info(f"Columnas finales: {tokenized_ds_dict['train'].column_names}")
    logging.info("--- Pipeline de Preparación de Datos Finalizado ---")

    return tokenized_ds_dict 