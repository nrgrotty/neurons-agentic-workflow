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

class RefinerOutput(BaseModel):
    editing_instructions: str
    index: int = 0
    reasoning: str = ""

class PlannerOutput(BaseModel):
    editing_instructions: str
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


class MetricRank(BaseModel):
    metric_name: str
    rank: int  # 1 = best, N = worst


class VariantRanking(BaseModel):
    variant_index: int
    ranks: list[MetricRank]


class EvaluatorRankings(BaseModel):
    variant_rankings: list[VariantRanking]
    reasoning: str


class AuditEntry(BaseModel):
    """A single recorded decision with its rationale."""
    node: str
    timestamp: datetime
    decision: str
    reasoning: str
    variant_index: int | None = None
    iteration: int | None = None


class EditorWorkerState(BaseModel):
    """State for a single parallel editor→critic→refiner loop."""
    image: Path
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines
    editing_instructions: str
    variant_index: int = 0
    edited_image: Path | None = None
    approved: bool | None = None
    critic_feedback: str | None = None
    iteration: int = 0
    audit_trail: Annotated[list[AuditEntry], operator.add] = []


class GraphState(BaseModel):
    image: Path
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines
    editing_instructions: str = ""
    evaluation_metrics: EvaluationMetrics | None = None
    edited_images: Annotated[list[Path], operator.add] = []
    final_image: Path | None = None
    audit_trail: Annotated[list[AuditEntry], operator.add] = []

class PipelineState(BaseModel):
    image: Path
    brand_guidelines: BrandGuidelines
    recommendations: list[Recommendation]
    recommendation_results: Annotated[list["RecommendationResult"], operator.add] = []
    audit_trail: Annotated[list[AuditEntry], operator.add] = []


class RecommendationResult(BaseModel):
    recommendation_id: str
    original_image: Path
    best_variant: Path
    other_variants: list[Path]

class PipelineOutput(BaseModel):
    recommendation_results: list[RecommendationResult]
    audit_trail: list[AuditEntry]