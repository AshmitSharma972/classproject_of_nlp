"""
Shared Pydantic schemas for the AI-Based Clinical Patient Case-Taking and History
Understanding System.

Every pipeline stage consumes and produces objects defined here.
All Span objects carry character offsets mapped to the ORIGINAL raw input text.
"""

from __future__ import annotations

import uuid
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


@dataclass
class PatientDemographics:
    """Report-only patient details; never passed to ClinicalPipeline."""
    full_name: str
    age: Optional[int]
    date_of_birth: Optional[date]
    sex: str
    patient_id: str
    phone: Optional[str]
    address_city: Optional[str]
    weight_kg: Optional[float]
    height_cm: Optional[float]
    blood_group: Optional[str]
    known_allergies: Optional[str]
    current_medications: Optional[str]
    emergency_contact_name: Optional[str]
    emergency_contact_phone: Optional[str]
    referring_physician: Optional[str]
    visit_date: date
    department: Optional[str]
    examining_physician: Optional[str]


class DemographicsInput(BaseModel):
    """Server-side validation for patient intake and explicit consent."""
    model_config = {"extra": "forbid"}
    full_name: str
    age: Optional[int] = Field(default=None, ge=0, le=120)
    date_of_birth: Optional[date] = None
    sex: Literal["Male", "Female", "Other", "Not stated"]
    patient_id: Optional[str] = Field(default=None, max_length=40)
    phone: Optional[str] = None
    address_city: Optional[str] = Field(default=None, max_length=300)
    weight_kg: Optional[float] = Field(default=None, gt=0, le=1000)
    height_cm: Optional[float] = Field(default=None, gt=0, le=300)
    blood_group: Optional[str] = Field(default=None, max_length=8)
    known_allergies: Optional[str] = Field(default=None, max_length=2000)
    current_medications: Optional[str] = Field(default=None, max_length=4000)
    emergency_contact_name: Optional[str] = Field(default=None, max_length=100)
    emergency_contact_phone: Optional[str] = None
    referring_physician: Optional[str] = Field(default=None, max_length=100)
    visit_date: date = Field(default_factory=date.today)
    department: Optional[str] = Field(default=None, max_length=150)
    examining_physician: Optional[str] = Field(default=None, max_length=100)
    consent: bool

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        value = value.strip()
        if not 2 <= len(value) <= 100:
            raise ValueError("full_name must contain 2 to 100 characters")
        return value

    @field_validator("phone", "emergency_contact_phone")
    @classmethod
    def validate_phone(cls, value: Optional[str]) -> Optional[str]:
        if value in (None, ""):
            return None
        if not re.fullmatch(r"[0-9]{7,15}", value):
            raise ValueError("phone must contain 7 to 15 digits")
        return value

    @model_validator(mode="after")
    def validate_visit_and_consent(self):
        if self.age is None and self.date_of_birth is None:
            raise ValueError("age or date_of_birth is required")
        if self.visit_date > date.today():
            raise ValueError("visit_date cannot be in the future")
        if self.date_of_birth:
            if self.date_of_birth > self.visit_date:
                raise ValueError("date_of_birth cannot be after visit_date")
            years = self.visit_date.year - self.date_of_birth.year
            if (self.visit_date.month, self.visit_date.day) < (self.date_of_birth.month, self.date_of_birth.day):
                years -= 1
            if not 0 <= years <= 120:
                raise ValueError("calculated age must be between 0 and 120")
            self.age = years
        if not self.consent:
            raise ValueError("consent is required")
        return self

    def to_patient_demographics(self) -> PatientDemographics:
        values = self.model_dump(exclude={"consent"})
        return PatientDemographics(**values)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    SYMPTOM = "SYMPTOM"
    DISEASE = "DISEASE"
    MEDICATION = "MEDICATION"
    BODY_PART = "BODY_PART"
    TEST = "TEST"
    TEST_RESULT = "TEST_RESULT"
    PROCEDURE = "PROCEDURE"
    DOSAGE = "DOSAGE"
    FREQUENCY = "FREQUENCY"
    DATE = "DATE"
    DURATION = "DURATION"
    SEVERITY = "SEVERITY"
    ALLERGEN = "ALLERGEN"
    FAMILY_MEMBER = "FAMILY_MEMBER"


class AssertionStatus(str, Enum):
    PRESENT = "PRESENT"
    NEGATED = "NEGATED"
    POSSIBLE = "POSSIBLE"
    HISTORICAL = "HISTORICAL"
    FAMILY = "FAMILY"


class Experiencer(str, Enum):
    PATIENT = "PATIENT"
    OTHER = "OTHER"


class RelationType(str, Enum):
    SYMPTOM_DURATION = "SYMPTOM-DURATION"
    SYMPTOM_SEVERITY = "SYMPTOM-SEVERITY"
    SYMPTOM_BODY_PART = "SYMPTOM-BODY_PART"
    MEDICATION_DOSAGE = "MEDICATION-DOSAGE"
    MEDICATION_FREQUENCY = "MEDICATION-FREQUENCY"
    DISEASE_STATUS = "DISEASE-STATUS"
    TEST_RESULT = "TEST-RESULT"
    PROCEDURE_DATE = "PROCEDURE-DATE"


class SectionType(str, Enum):
    CHIEF_COMPLAINT = "Chief Complaint"
    HPI = "HPI"
    PAST_HISTORY = "Past History"
    DRUG_HISTORY = "Drug History"
    ALLERGY = "Allergy"
    FAMILY_HISTORY = "Family History"
    PERSONAL_HISTORY = "Personal History"
    ROS = "ROS"


class TimeExprType(str, Enum):
    DURATION = "DURATION"
    ONSET = "ONSET"
    DATE = "DATE"
    RELATIVE = "RELATIVE"
    TREATMENT_DATE = "TREATMENT_DATE"
    PROCEDURE_DATE = "PROCEDURE_DATE"
    PREVIOUS_EPISODE = "PREVIOUS_EPISODE"


class ReviewActionType(str, Enum):
    ACCEPT = "ACCEPT"
    EDIT = "EDIT"
    REJECT = "REJECT"
    VERIFY = "VERIFY"


# ---------------------------------------------------------------------------
# Core span -- ALL offsets are into the ORIGINAL raw input text
# ---------------------------------------------------------------------------

class Span(BaseModel):
    """Character-level span in the ORIGINAL raw input text."""
    text: str = Field(..., description="The verbatim text from the original input")
    start: int = Field(..., ge=0, description="Start character offset in original raw text (inclusive)")
    end: int = Field(..., ge=0, description="End character offset in original raw text (exclusive)")
    source_sentence_id: int = Field(
        ..., ge=0,
        description="0-based index of the sentence this span belongs to"
    )

    @model_validator(mode="after")
    def start_before_end(self) -> "Span":
        if self.start > self.end:
            raise ValueError(f"Span start ({self.start}) must be <= end ({self.end})")
        return self

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Named entity
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A single clinical named entity with its provenance span."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique entity ID")
    span: Span
    type: EntityType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence [0, 1]")
    normalized_text: Optional[str] = Field(
        None, description="Canonical / normalized form of the entity text"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Assertion / Negation
# ---------------------------------------------------------------------------

class Assertion(BaseModel):
    """Assertion context applied to a named entity."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = Field(..., description="ID of the Entity this assertion applies to")
    status: AssertionStatus = AssertionStatus.PRESENT
    experiencer: Experiencer = Experiencer.PATIENT
    trigger_text: Optional[str] = Field(
        None, description="The trigger word/phrase that caused this assertion, e.g. 'denies'"
    )
    trigger_span: Optional[Span] = Field(
        None, description="Span of the trigger in the original raw text"
    )
    confidence: float = Field(1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Temporal expression
# ---------------------------------------------------------------------------

class TimeExpr(BaseModel):
    """A temporal expression linked to one or more entities."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    span: Span
    type: TimeExprType
    normalized_value: Optional[str] = Field(
        None, description="Normalized representation, e.g. 'P5D' (ISO 8601) or '5 days'"
    )
    raw_value: str = Field(..., description="Verbatim text of the temporal expression")
    linked_entity_ids: List[str] = Field(
        default_factory=list,
        description="IDs of entities this temporal expression modifies"
    )
    timeline_anchor: Optional[str] = Field(
        None, description="Absolute or relative anchor point for timeline ordering"
    )


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------

class Relation(BaseModel):
    """A directed relation between two entities."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    head_entity_id: str = Field(..., description="ID of the head (source) entity")
    tail_entity_id: str = Field(..., description="ID of the tail (target) entity")
    relation_type: RelationType
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_sentence_id: int = Field(..., ge=0)
    evidence_text: Optional[str] = Field(
        None, description="The sentence text that supports this relation"
    )


# ---------------------------------------------------------------------------
# Section item -- an entity assigned to a clinical section
# ---------------------------------------------------------------------------

class SectionItem(BaseModel):
    """An entity assigned to a clinical note section with full provenance."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str
    section: SectionType
    assertion_id: Optional[str] = None
    temporal_ids: List[str] = Field(default_factory=list)
    relation_ids: List[str] = Field(default_factory=list)
    provenance_span: Span = Field(
        ..., description="Span of the entity in the original raw text"
    )
    display_text: Optional[str] = Field(
        None, description="Human-readable label for this item in the clinical summary"
    )


# ---------------------------------------------------------------------------
# Timeline entry
# ---------------------------------------------------------------------------

class TimelineEntry(BaseModel):
    """A single entry in the reconstructed clinical timeline."""
    order: int = Field(..., description="Relative ordering position (lower = earlier)")
    label: str = Field(..., description="Human-readable label for this timeline point")
    time_expr_id: Optional[str] = None
    linked_entity_ids: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Patient record -- the unified structured output of the pipeline
# ---------------------------------------------------------------------------

class PatientRecord(BaseModel):
    """
    The complete structured representation of a patient's clinical history.

    Every clinically meaningful item retains:
    - Source sentence index
    - Original character offsets
    - Confidence
    - Relevant assertion / temporal information
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str = Field(..., description="The verbatim original patient narration text")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Core NLP outputs
    entities: List[Entity] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    time_expressions: List[TimeExpr] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)

    # Section-level organization
    section_items: List[SectionItem] = Field(default_factory=list)

    # Reconstructed timeline
    timeline: List[TimelineEntry] = Field(default_factory=list)

    # Grouped by section for easy rendering
    sections: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Section name → list of display strings for that section"
    )

    # Metadata about the preprocessing step
    preprocessing_metadata: Dict[str, Any] = Field(default_factory=dict)

    # ---- Convenience accessors ----------------------------------------

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None

    def get_assertion_for_entity(self, entity_id: str) -> Optional[Assertion]:
        for a in self.assertions:
            if a.entity_id == entity_id:
                return a
        return None

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        return [e for e in self.entities if e.type == entity_type]

    def get_section_entities(self, section: SectionType) -> List[SectionItem]:
        return [s for s in self.section_items if s.section == section]


# ---------------------------------------------------------------------------
# Physician review action
# ---------------------------------------------------------------------------

class ReviewAction(BaseModel):
    """A single physician review action on a clinical record item."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    item_id: str = Field(..., description="ID of the item being reviewed (entity, section_item, etc.)")
    action: ReviewActionType
    old_value: Optional[str] = Field(None, description="Original value before edit (for EDIT actions)")
    new_value: Optional[str] = Field(None, description="New value after edit (for EDIT actions)")
    reviewer: str = Field(..., description="Identifier of the reviewing physician/user")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    note: Optional[str] = Field(None, description="Optional free-text note from the reviewer")

    DISCLAIMER: str = Field(
        default="Decision support only. The physician remains responsible for clinical decisions.",
        description="Mandatory disclaimer text",
        frozen=True,
    )


# ---------------------------------------------------------------------------
# Pipeline-level preprocessing output (used between stages)
# ---------------------------------------------------------------------------

class SentenceInfo(BaseModel):
    """Metadata for a single sentence produced by the preprocessor."""
    sentence_id: int = Field(..., ge=0)
    text: str = Field(..., description="Cleaned/normalized sentence text")
    raw_start: int = Field(..., ge=0, description="Start offset in original raw text")
    raw_end: int = Field(..., ge=0, description="End offset in original raw text (exclusive)")
    tokens: List[str] = Field(default_factory=list)

    model_config = {"frozen": False}


class PreprocessingOutput(BaseModel):
    """Complete output of the preprocessing stage."""
    raw_text: str
    normalized_text: str
    sentences: List[SentenceInfo] = Field(default_factory=list)
    # Character-level offset map: normalized_char_index → raw_char_index
    offset_map: Dict[int, int] = Field(
        default_factory=dict,
        description="Maps each character index in normalized_text to the corresponding index in raw_text"
    )
    spell_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of suggested spelling corrections (flagged, never silently applied)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
