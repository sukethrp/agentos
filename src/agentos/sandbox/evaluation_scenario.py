from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class EvaluationScenario(BaseModel):
    scenario_id: str
    input: str
    expected_output: str
    rubric: str
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> list["EvaluationScenario"]:
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        if data is None:
            return []
        if isinstance(data, list):
            return [cls(**item) for item in data]
        if isinstance(data, dict):
            if "scenarios" in data:
                return [cls(**item) for item in data["scenarios"]]
            return [cls(**data)]
        return []
