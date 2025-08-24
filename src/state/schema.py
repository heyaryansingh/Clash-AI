from dataclasses import dataclass
from typing import List, Dict

@dataclass
class GameState:
    hand: List[str]
    elixir: float
    towers: Dict[str,int]
    units: List
    time_s: int
    history: List
