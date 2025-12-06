"""
FastAPI 응답 모델 (Pydantic 스키마)
"""

from pydantic import BaseModel, Field, field_validator


class NutritionInfo(BaseModel):
    """영양소 정보"""

    calories_kcal: float = Field(..., ge=0, description="열량 (kcal)")
    protein_g: float = Field(..., ge=0, description="단백질 (g)")
    fat_g: float = Field(..., ge=0, description="지방 (g)")
    carbs_g: float = Field(..., ge=0, description="탄수화물 (g)")

    @field_validator('*')
    @classmethod
    def round_to_two_decimals(cls, v):
        """소수점 2자리로 반올림"""
        return round(v, 2)


class AnalysisResponse(BaseModel):
    """음식 분석 결과"""

    food_name: str = Field(..., description="음식 이름")
    confidence: float = Field(..., ge=0, le=1, description="분류 신뢰도 (0~1)")
    volume_ml: float = Field(..., ge=0, description="부피 (ml)")
    mass_g: float = Field(..., ge=0, description="질량 (g)")
    nutrition: NutritionInfo

    @field_validator('confidence', 'volume_ml', 'mass_g')
    @classmethod
    def round_values(cls, v):
        """소수점 2자리로 반올림"""
        return round(v, 2)

    class Config:
        json_schema_extra = {
            "example": {
                "food_name": "비빔밥",
                "confidence": 0.94,
                "volume_ml": 350.50,
                "mass_g": 350.50,
                "nutrition": {
                    "calories_kcal": 525.75,
                    "protein_g": 28.04,
                    "fat_g": 17.53,
                    "carbs_g": 87.63
                }
            }
        }


class ErrorResponse(BaseModel):
    """에러 응답"""

    detail: str = Field(..., description="에러 메시지")
