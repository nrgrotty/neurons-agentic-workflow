import operator
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

class CreativeEditorInput(BaseModel):
    image: Path
    brand_guidelines: BrandGuidelines
    recommendations: list[Recommendation]

class PlannerInput(BaseModel):
    brand_guidelines: BrandGuidelines
    recommendation: Recommendation

class SubTask(BaseModel):
    description: str


class PlannerOutput(BaseModel):
    subtasks: list[SubTask]

class EditorInput(BaseModel):
    image: Path
    subtask: SubTask


class EditorOutput(BaseModel):
    edited_image: Path


class CriticInput(BaseModel):
    edited_image: Path
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines

class CriticOutput(BaseModel):
    approval: bool
    feedback: str


class EvaluationMetric(BaseModel):
    name: str
    description: str
    weight: float  # 0.0–1.0, relative importance


class EvaluationMetrics(BaseModel):
    metrics: list[EvaluationMetric]


class WorkerState(BaseModel):
    """State for a single parallel editor→critic→refiner loop."""
    image: Path
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines
    subtask: SubTask
    subtask_index: int = 0
    edited_image: Path | None = None
    approved: bool | None = None
    critic_feedback: str | None = None
    iteration: int = 0


class GraphState(BaseModel):
    image: Path
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines
    subtasks: list[SubTask] = []
    evaluation_metrics: EvaluationMetrics | None = None
    edited_images: Annotated[list[Path], operator.add] = []
    final_image: Path | None = None


class PipelineState(BaseModel):
    image: Path
    brand_guidelines: BrandGuidelines
    recommendations: list[Recommendation]
    final_images: Annotated[list[Path], operator.add] = []