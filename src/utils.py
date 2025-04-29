import logging
from typing import List

from .data_models import BoundaryAnnotation, Entry
from .splitters import EntrySplitter

# from transformers import PreTrainedTokenizerBase # Evitar dependencia directa si es posible

def split_entry(b_annotations_list: List[List[BoundaryAnnotation]], tokenizer, max_length_preset=None) -> (bool, List[List[BoundaryAnnotation]]):
   """Divide recursivamente listas de BoundaryAnnotations si el texto tokenizado excede el límite.

   Args:
       b_annotations_list: Lista de listas, donde cada sublista representa un segmento potencial.
       tokenizer: Tokenizador de Hugging Face.
       max_length_preset: Límite máximo opcional de tokens (por defecto, tokenizer.model_max_length).

   Returns:
       Tuple (bool, List[List[BoundaryAnnotation]]):
           - bool: True si todas las sublistas finales respetan el límite, False si hubo divisiones.
           - List[List[BoundaryAnnotation]]: Lista final de sublistas de anotaciones, divididas si fue necesario.
   """
   res = []
   rdy = True
   length_limit = tokenizer.model_max_length if max_length_preset is None else max_length_preset

   processed_list = []

   for b_annotations in b_annotations_list:
       if not b_annotations: continue

       spans_to_join = [str(ba.span) for ba in b_annotations if ba.span is not None]
       if not spans_to_join:
           logging.warning("Lista de BoundaryAnnotations sin spans válidos encontrada.")
           processed_list.append(b_annotations)
           continue

       joined_text = " ".join(spans_to_join)

       try:
            tokenized_ids = tokenizer.encode(joined_text, truncation=False, add_special_tokens=False)
       except Exception as e:
            logging.error(f"Fallo al tokenizar texto de {len(b_annotations)} anotaciones: {e}")
            rdy = False
            continue

       if len(tokenized_ids) > length_limit:
           logging.debug(f"Entrada necesita división: {len(tokenized_ids)} tokens > límite {length_limit}. Anotaciones: {len(b_annotations)}.")
           rdy = False

           max_annotations = length_limit
           if len(b_annotations) > 0:
               avg_tokens_per_annotation = len(tokenized_ids) / len(b_annotations)
               if avg_tokens_per_annotation > 0:
                   max_annotations = int(length_limit / avg_tokens_per_annotation)
               else:
                   max_annotations = len(b_annotations) // 2
           max_annotations = max(max_annotations, 1)
           min_split_size = max(1, max_annotations // 4)

           logging.debug(f"Intentando dividir {len(b_annotations)} anotaciones en chunks <= {max_annotations}, min_size {min_split_size}")
           esplitter = EntrySplitter(max_size=max_annotations, min_size=min_split_size)
           splits = esplitter.split(b_annotations)

           if len(splits) == 1 and splits[0] == b_annotations:
               logging.error(f"La división no redujo el tamaño para la lista de {len(b_annotations)} anotaciones. Deteniendo recursión.")
               processed_list.append(b_annotations)
               continue

           recursive_ready, fine_splits = split_entry(splits, tokenizer, max_length_preset)
           processed_list.extend(fine_splits)
           if not recursive_ready:
                rdy = False

       else:
           processed_list.append(b_annotations)

   return rdy, processed_list

def get_text_splits(entry: Entry, ba_splits: List[List[BoundaryAnnotation]]) -> List[str]:
   """Extrae los segmentos de texto correspondientes a cada sublista de BoundaryAnnotations."""
   result = []
   for b_ann_list in ba_splits:
        if not b_ann_list:
             logging.debug("Omitiendo lista vacía de BoundaryAnnotation en get_text_splits.")
             continue

        try:
            start_offset = b_ann_list[0].start_offset
            end_offset = b_ann_list[-1].end_offset
        except (AttributeError, TypeError, IndexError) as e:
             logging.error(f"Error accediendo a offsets en lista BoundaryAnnotation: {e}. Contenido: {b_ann_list}")
             continue

        if start_offset >= 0 and end_offset >= start_offset and end_offset < len(entry.note_text):
             text_segment = entry.note_text[start_offset : end_offset + 1]
             result.append(text_segment)
        else:
             logging.warning(f"Offsets inválidos o fuera de rango: start={start_offset}, end={end_offset}, note_len={len(entry.note_text)}. Omitiendo split.")

   return result

def getBoudaryAnnotationsForRange(bas: List[BoundaryAnnotation], start: int, end: int) -> List[BoundaryAnnotation]:
   """Filtra BoundaryAnnotations cuyo inicio cae dentro del rango de caracteres [start, end)."""
   result = []
   if not isinstance(bas, list):
       logging.warning("La entrada 'bas' no es una lista en getBoudaryAnnotationsForRange.")
       return []

   for ba in bas:
        try:
            if ba.start_offset >= start and ba.start_offset < end:
                 result.append(ba)
        except AttributeError:
             logging.warning(f"Item {ba} sin atributo 'start_offset'.")
        except Exception as e:
            logging.error(f"Error procesando boundary annotation {ba}: {e}")

   return result 