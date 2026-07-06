import json
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def save_json(model: BaseModel, prefix: str) -> Path:
    """Save any Pydantic model as timestamped JSON in data/.

    Example: save_json(research_output, "research") ->
             data/research_20260706_142301.json

    Returns the path so the caller can log/print it or pass it along.
    """
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"{prefix}_{timestamp}.json"
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_json(path: Path, model_cls: type[BaseModel]) -> BaseModel:
    """Load a saved JSON file back into its Pydantic model.

    model_cls is the class itself (e.g. ResearchOutput), not an instance —
    that's what lets this function stay generic instead of writing a
    separate load_research(), load_draft(), load_critic_report() each time.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return model_cls.model_validate(data)