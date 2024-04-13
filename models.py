from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class Layer(BaseModel):
    name: str
    params: Dict[str, Any]


class Layers(BaseModel):
    name: str
    user_id: str
    layers: List[Layer]
    activation_function: Optional[str] = None


class Train(BaseModel):
    batch_size: int
    learning_rate: float
    optimizer: str
    epochs: int
    user_id: int
    model_name: int
