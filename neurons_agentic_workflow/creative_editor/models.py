from enum import Enum

from pydantic import BaseModel


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

class PlannerInput(BaseModel):
    image: str
    brand_guidelines: BrandGuidelines
    recommendation: Recommendation

class SubTask(BaseModel):
    image:str 
    description: str

class EditorInput(BaseModel):
    image: str
    sub_task: SubTask

class CriticInput(BaseModel):
    edited_image: str
    recommendation: Recommendation
    brand_guidelines: BrandGuidelines

class CriticOutput(BaseModel):
    approval: bool
    sub_task: SubTask