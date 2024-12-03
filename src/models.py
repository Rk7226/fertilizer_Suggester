from pydantic import BaseModel, Field

class FertilizerInput(BaseModel):
    Soil_Type: str = Field(..., description="Type of soil")
    Crop_Type: str = Field(..., description="Type of crop")
    Nitrogen: float = Field(..., description="Nitrogen level")
    Phosphorus: float = Field(..., description="Phosphorus level")
    Potassium: float = Field(..., description="Potassium level")
    Temperature: float = Field(..., description="Temperature")
    Humidity: float = Field(..., description="Humidity")
    Moisture: float = Field(..., description="Moisture level")