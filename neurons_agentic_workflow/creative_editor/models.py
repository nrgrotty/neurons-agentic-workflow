from enum import Enum

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
    recommendation: Recommendation

class PlannerInput(BaseModel):
    brand_guidelines: BrandGuidelines
    recommendation: Recommendation

class SubTask(BaseModel):
    description: str

class EditorInput(BaseModel):
    image: Path
    sub_task: SubTask


class EditorOutput(BaseModel):
    edited_image: Path


class CriticInput(BaseModel):
    edited_image: Path
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines

class CriticOutput(BaseModel):
    approval: bool
    sub_task: SubTask