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


class CreativeFeedback(BaseModel):
    filename: str
    brand_guidelines: BrandGuidelines
    recommendations: list[Recommendation]


class Tool(str, Enum):
    PATCH = "patch"
    STYLE = "style"
    WRITE = "write"


class SubTask(BaseModel):
    tool: Tool
    summary: str
