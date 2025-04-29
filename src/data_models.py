import json
from enum import Enum
from typing import List, Any
from pydantic import BaseModel


class ClinicalSections(str, Enum):
    """Enumeración de las secciones clínicas definidas en ClinAIS."""
    PRESENT_ILLNESS = "PRESENT_ILLNESS"
    DERIVED_FROM_TO = "DERIVED_FROM/TO"
    PAST_MEDICAL_HISTORY = "PAST_MEDICAL_HISTORY"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    EXPLORATION = "EXPLORATION"
    TREATMENT = "TREATMENT"
    EVOLUTION = "EVOLUTION"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class SectionAnnotation(BaseModel):
    """Representa una anotación de sección completa."""
    segment: str
    label: ClinicalSections
    start_offset: int
    end_offset: int


class SectionAnnotations(BaseModel):
    """Contiene las anotaciones de sección gold y predichas."""
    gold: List[SectionAnnotation] = []
    prediction: List[SectionAnnotation] = []


class BoundaryAnnotation(BaseModel):
    """Representa una anotación de límite (boundary)."""
    span: str
    boundary: ClinicalSections | None # Puede no haber boundary al final
    start_offset: int
    end_offset: int


class BoundaryAnnotations(BaseModel):
    """Contiene las anotaciones de límite gold y predichas."""
    gold: List[BoundaryAnnotation] = []
    prediction: List[BoundaryAnnotation] = []


class Entry(BaseModel):
    """Representa una entrada (nota clínica) en el dataset."""
    note_id: str
    note_text: str
    section_annotation: SectionAnnotations = SectionAnnotations()
    boundary_annotation: BoundaryAnnotations = BoundaryAnnotations()

    def toJson(self):
        # Ensure default handles Enum correctly if needed, though Pydantic usually does
        return json.dumps(self.dict(by_alias=True), ensure_ascii=False) # Use .dict()


class ClinAISDataset(BaseModel):
    """Representa el dataset ClinAIS completo."""
    annotated_entries: dict[str, Entry]
    scores: dict[str, Any] = {}

    def getEntry(self, idx: int) -> Entry:
        key = list(self.annotated_entries.keys())[idx]
        return self.annotated_entries[key]

    def toJson(self):
         # Use .dict() for proper Pydantic serialization, handle Enums
        return json.dumps(self.dict(by_alias=True), ensure_ascii=False)

# --- Models used in Postprocessing/Prediction ---

class PredictionSection(BaseModel):
   """Representa una sección predicha por el modelo (formato pipeline HF)."""
   entity_group: ClinicalSections
   score: float | None = None
   word: str # Puede ser un token o un grupo de tokens si se agrega
   start: int
   end: int

   class Config:
       use_enum_values = True # Ensure Enum values are used if needed


class Prediction(BaseModel):
    """Contenedor para las secciones predichas de una entrada."""
    sections: List[PredictionSection] = [] 