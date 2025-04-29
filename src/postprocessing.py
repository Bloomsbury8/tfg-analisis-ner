import logging
from typing import List
from enum import Enum

# Use relative import for data models within the same package
from .data_models import Prediction, PredictionSection, ClinicalSections

class PredictionPostProcessor:
   """Clase para aplicar pasos de post-procesado a las predicciones de secciones.

   Modifica el objeto Prediction internamente.
   """
   def __init__(self, prediction: Prediction, min_section_size=3, verbose=False):
       # Operar sobre una copia para evitar modificar el objeto original
       self.prediction = prediction.copy(deep=True) if hasattr(prediction, 'copy') else self._manual_copy(prediction)
       self.min_section_size = min_section_size
       self.punctuation_marks = [',', '.', ';', ':', ')', ']', '}', '!', '?']
       self.verbose = verbose

   def _manual_copy(self, prediction: Prediction) -> Prediction:
        """Copia manual básica si .copy() no está disponible (Pydantic v1)."""
        new_sections = [self._manual_copy_section(sec) for sec in prediction.sections]
        return Prediction(sections=new_sections)

   def _manual_copy_section(self, section: PredictionSection) -> PredictionSection:
        """Copia manual de una PredictionSection."""
        # Asume que PredictionSection es Pydantic o tiene .dict()
        try:
            return PredictionSection(**section.dict())
        except AttributeError:
             # Fallback si no es Pydantic o no tiene .dict()
             logging.error("No se pudo copiar PredictionSection manualmente, ¿falta .dict()?")
             # Devolver una copia simple (puede ser superficial)
             import copy
             return copy.copy(section)

   def get_section_size(self, sec: PredictionSection):
       """Calcula el número de palabras en el texto de una sección."""
       return len(sec.word.strip().split()) if sec.word else 0

   def merge_undersized_sections(self):
       """Fusiona secciones más pequeñas que min_section_size con adyacentes."""
       if not self.prediction.sections:
            return self.prediction

       processed_sections: List[PredictionSection] = []
       i = 0
       while i < len(self.prediction.sections):
            current_sec = self.prediction.sections[i]
            current_size = self.get_section_size(current_sec)

            if current_size < self.min_section_size:
                merged = False
                # Intentar fusionar con la sección *anterior* ya procesada
                if processed_sections:
                    prev_sec = processed_sections[-1]
                    # Nota TFG: Considerar si añadir chequeo de compatibilidad (ej: mismo label?)
                    if self.verbose: logging.debug(f"PP: Fusionando sección corta {i} ('{current_sec.word[:20]}...', {current_size}) con anterior.")
                    prev_sec.word = (prev_sec.word + " " + current_sec.word).strip()
                    prev_sec.end = current_sec.end
                    prev_sec.score = None # El score se vuelve ambiguo
                    merged = True
                # Intentar fusionar con la sección *siguiente*
                elif i + 1 < len(self.prediction.sections):
                    next_sec = self.prediction.sections[i+1]
                    # Nota TFG: Considerar chequeo de compatibilidad
                    if self.verbose: logging.debug(f"PP: Fusionando sección corta {i} ('{current_sec.word[:20]}...', {current_size}) con siguiente {i+1}.")
                    # Crear copia de next_sec para modificarla
                    modified_next_sec = self._manual_copy_section(next_sec)
                    modified_next_sec.word = (current_sec.word + " " + next_sec.word).strip()
                    modified_next_sec.start = current_sec.start
                    modified_next_sec.score = None
                    processed_sections.append(modified_next_sec)
                    i += 1 # Saltar la siguiente sección original en la próxima iteración
                    merged = True
                else:
                     # Sección corta al final, no se puede fusionar
                     if self.verbose: logging.debug(f"PP: Manteniendo sección corta {i} ('{current_sec.word[:20]}...', {current_size}) al no poder fusionar.")
                     processed_sections.append(current_sec)
            else:
                # Sección con tamaño suficiente
                processed_sections.append(current_sec)

            i += 1

       self.prediction.sections = processed_sections
       return self.prediction

   def reassign_leading_punctuation_marks(self):
      """Mueve signos de puntuación iniciales de una sección al final de la anterior."""
      if len(self.prediction.sections) < 2:
           return

      for i in range(1, len(self.prediction.sections)):
           current_sec = self.prediction.sections[i]
           prev_sec = self.prediction.sections[i-1]

           moved_punctuation = ""
           original_word = current_sec.word
           original_start = current_sec.start

           # Acumular puntuación inicial
           while current_sec.word and current_sec.word[0] in self.punctuation_marks:
                punc = current_sec.word[0]
                moved_punctuation += punc
                current_sec.word = current_sec.word[1:].lstrip()
                # Ajustar offset por cada carácter movido
                current_sec.start += len(punc)

           if moved_punctuation:
                if self.verbose: logging.debug(f"PP: Moviendo puntuación '{moved_punctuation}' de inicio sección {i} a fin sección {i-1}.")
                prev_sec.word += moved_punctuation
                prev_sec.end += len(moved_punctuation)

      # Eliminar secciones que quedaron vacías
      self.prediction.sections = [sec for sec in self.prediction.sections if sec.word and not sec.word.isspace()]
      return self.prediction

   def merge_contiguous_equivalent_sections(self):
      """Fusiona secciones adyacentes con la misma etiqueta (entity_group)."""
      if len(self.prediction.sections) < 2:
           return self.prediction

      merged_sections: List[PredictionSection] = []
      if not self.prediction.sections: return self.prediction # Comprobación extra

      current_merged_section = self._manual_copy_section(self.prediction.sections[0])

      for i in range(1, len(self.prediction.sections)):
           next_sec = self.prediction.sections[i]

           if next_sec.entity_group == current_merged_section.entity_group:
                if self.verbose: logging.debug(f"PP: Fusionando sección {i} ({next_sec.entity_group}) con anterior.")
                # Combinar palabras asumiendo espacio intermedio
                current_merged_section.word = (current_merged_section.word + " " + next_sec.word).strip()
                current_merged_section.end = next_sec.end
                current_merged_section.score = None # Score ambiguo
           else:
                # Finalizar sección fusionada actual y empezar una nueva
                merged_sections.append(current_merged_section)
                current_merged_section = self._manual_copy_section(next_sec)

      merged_sections.append(current_merged_section) # Añadir la última

      self.prediction.sections = merged_sections
      return self.prediction

   def do_all(self, verbose=None) -> Prediction:
      """Aplica todos los pasos de post-procesado en secuencia."""
      if verbose is not None: self.verbose = verbose

      if self.verbose: logging.info("--- Iniciando Post-procesado de Predicciones --- ")
      original_count = len(self.prediction.sections)
      if self.verbose: logging.debug(f"Secciones iniciales: {original_count}")

      # Orden de aplicación definido
      self.merge_undersized_sections()
      if self.verbose: logging.debug(f"Tras fusionar cortas: {len(self.prediction.sections)} secciones")

      self.reassign_leading_punctuation_marks()
      if self.verbose: logging.debug(f"Tras reasignar puntuación: {len(self.prediction.sections)} secciones")

      self.merge_contiguous_equivalent_sections()
      if self.verbose: logging.debug(f"Tras fusionar contiguas: {len(self.prediction.sections)} secciones")

      final_count = len(self.prediction.sections)
      if self.verbose: logging.info(f"--- Post-procesado finalizado. Secciones: {original_count} -> {final_count} --- ")

      return self.prediction 