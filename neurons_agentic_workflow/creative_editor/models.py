import operator
from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel
from pathlib import Path

class BrandGuidelines(BaseModel):
    protected_regions: list[str]
    typography: str
    aspect_ratio: str
    brand_elements: str


class RecommendationType(str, Enum):
    COLOUR_MOOD = "colour_mood"
    COPY_MESSAGING = "copy_messaging"
    CONTRAST_SALIENCE = "contrast_salience"
    COMPOSITION = "composition"


class Recommendation(BaseModel):
    id: str
    title: str
    description: str
    type: RecommendationType

class PipelineInput(BaseModel):
    image: Path
    brand_guidelines: BrandGuidelines
    recommendations: list[Recommendation]

class SubTask(BaseModel):
    description: str
    index: int = 0
    reasoning: str = ""

class PlannerOutput(BaseModel):
    subtasks: list[SubTask]
    reasoning: str

class EditorOutput(BaseModel):
    edited_image: Path
class CriticOutput(BaseModel):
    approval: bool
    feedback: str
    reasoning: str


class EvaluationMetric(BaseModel):
    name: str
    description: str
    weight: float


class EvaluationMetrics(BaseModel):
    metrics: list[EvaluationMetric]
    reasoning: str


class AuditEntry(BaseModel):
    """A single recorded decision with its rationale."""
    node: str
    timestamp: datetime
    decision: str
    reasoning: str
    subtask_index: int | None = None
    iteration: int | None = None


class EditorWorkerState(BaseModel):
    """State for a single parallel editor→critic→refiner loop."""
    image: Path
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines
    subtask: SubTask
    edited_image: Path | None = None
    approved: bool | None = None
    critic_feedback: str | None = None
    iteration: int = 0
    audit_trail: Annotated[list[AuditEntry], operator.add] = []


class GraphState(BaseModel):
    image: Path
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines
    subtasks: list[SubTask] = []
    evaluation_metrics: EvaluationMetrics | None = None
    edited_images: Annotated[list[Path], operator.add] = []
    final_image: Path | None = None
    audit_trail: Annotated[list[AuditEntry], operator.add] = []


class PipelineFormInput(BaseModel):
    brand_guidelines: BrandGuidelines
    recommendations: list[Recommendation]

class PipelineState(BaseModel):
    image: Path
    brand_guidelines: BrandGuidelines
    recommendations: list[Recommendation]
    final_images: Annotated[list[Path], operator.add] = []
    audit_trail: Annotated[list[AuditEntry], operator.add] = []


class PipelineOutput(BaseModel):
    final_images: list[EditorOutput]
    audit_trail: list[AuditEntry]