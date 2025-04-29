import logging
from typing import List

# Nota TFG: Splitters adaptados para dividir listas de palabras o anotaciones
# basándose en caracteres de puntuación y tamaño mínimo/máximo.

class WordListSplitter():
    """Divide una lista de strings (palabras/tokens) en sublistas."""
    def __init__(self, max_size: int, min_size: int, split_caracter_list=['.', ',', ';', ':']):
        self.max_size = max_size
        self.min_size = min_size
        self.split_caracters = split_caracter_list

    def get_split_index(self, spans: List[str], caracter='.'):
        """Encuentra el índice de división óptimo basado en un caracter y la posición media."""
        positions = []
        for idx, span in enumerate(spans):
            if span and isinstance(span, str) and span.endswith(caracter):
                if idx < len(spans) - 1 and spans[idx+1] and isinstance(spans[idx+1], str) and spans[idx+1].startswith('\n'):
                    positions.append(idx+1)
                else:
                    positions.append(idx)

        if not positions:
             return -1

        avg = sum(positions) / len(positions)
        return min(positions, key=lambda x: abs(x - avg))

    def split(self, spans: List[str], split_caracter_index=0):
        """Divide recursivamente la lista de spans."""
        if len(spans) <= self.max_size:
            return [spans]

        spos = self.get_split_index(spans, self.split_caracters[split_caracter_index])

        valid_spos = (spos != -1 and spos >= self.min_size and (len(spans) - spos) >= self.min_size)

        if not valid_spos:
            if split_caracter_index < len(self.split_caracters) - 1:
                logging.debug(f"WordListSplitter: Split inválido ({spos}) con '{self.split_caracters[split_caracter_index]}'. Probando siguiente.")
                return self.split(spans, split_caracter_index + 1)
            else:
                logging.warning(f"WordListSplitter: Caracteres agotados/split inválido ({spos}). Dividiendo {len(spans)} spans por la mitad.")
                spos = len(spans) // 2
                if spos == 0 and len(spans) > 1: spos = 1
                if spos < self.min_size or (len(spans) - spos) < self.min_size:
                    logging.warning(f"WordListSplitter: División forzada en {spos} viola min_size ({self.min_size}). Procediendo.")
                    if spos < self.min_size: spos = self.min_size

        if spos <= 0 or spos >= len(spans):
            logging.error(f"WordListSplitter: Error crítico - spos={spos} fuera de límites para len={len(spans)}.")
            return [spans]

        left = spans[:spos]
        right = spans[spos:]

        result = []
        if left: result.extend(self.split(left, 0))
        if right: result.extend(self.split(right, 0))

        if not result and spans:
             logging.error("WordListSplitter: Fallo inesperado en split. Devolviendo lista original.")
             return [spans]

        return result


class EntrySplitter():
    """Divide una lista de BoundaryAnnotation en sublistas."""
    from .data_models import BoundaryAnnotation

    def __init__(self, max_size: int, min_size: int, split_caracter_list=['.', ',', ';', ':']):
        self.max_size = max_size
        self.min_size = min_size
        self.split_caracters = split_caracter_list

    def get_split_index(self, b_annotations: List[BoundaryAnnotation], caracter='.'):
        """Encuentra el índice de división óptimo basado en caracter y posición media."""
        positions = []
        for idx, ba in enumerate(b_annotations):
            if ba.span and isinstance(ba.span, str) and ba.span.endswith(caracter):
                is_followed_by_newline = False
                if idx < len(b_annotations) - 1:
                    next_ba = b_annotations[idx+1]
                    if next_ba.span and isinstance(next_ba.span, str) and next_ba.span.startswith('\n'):
                        is_followed_by_newline = True

                positions.append(idx + 1)

        if not positions:
            return -1

        avg = sum(positions) / len(positions)
        return min(positions, key=lambda x: abs(x - avg))

    def split(self, b_annotations: List[BoundaryAnnotation], split_caracter_index=0) -> List[List[BoundaryAnnotation]]:
        """Divide recursivamente la lista de anotaciones."""
        if len(b_annotations) <= self.max_size:
            return [b_annotations]

        spos = self.get_split_index(b_annotations, self.split_caracters[split_caracter_index])

        valid_spos = (spos != -1 and spos >= self.min_size and (len(b_annotations) - spos) >= self.min_size)

        if not valid_spos:
            if split_caracter_index < len(self.split_caracters) - 1:
                logging.debug(f"EntrySplitter: Split inválido ({spos}) con '{self.split_caracters[split_caracter_index]}'. Probando siguiente.")
                return self.split(b_annotations, split_caracter_index + 1)
            else:
                logging.warning(f"EntrySplitter: Caracteres agotados/split inválido ({spos}). Dividiendo {len(b_annotations)} anotaciones por la mitad.")
                spos = len(b_annotations) // 2
                if spos == 0 and len(b_annotations) > 1: spos = 1
                if spos < self.min_size or (len(b_annotations) - spos) < self.min_size:
                    logging.warning(f"EntrySplitter: División forzada en {spos} viola min_size ({self.min_size}). Ajustando.")
                    if spos < self.min_size: spos = self.min_size

        if spos <= 0 or spos >= len(b_annotations):
            logging.error(f"EntrySplitter: Error crítico - spos={spos} fuera de límites para len={len(b_annotations)}.")
            return [b_annotations]

        left = b_annotations[:spos]
        right = b_annotations[spos:]

        result = []
        if left: result.extend(self.split(left, 0))
        if right: result.extend(self.split(right, 0))

        if not result and b_annotations:
             logging.error("EntrySplitter: Fallo inesperado en split. Devolviendo lista original.")
             return [b_annotations]

        return result 