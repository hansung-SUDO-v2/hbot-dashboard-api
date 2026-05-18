from dataclasses import dataclass, field
from typing import List, Optional

from bertopic import BERTopic


@dataclass
class AppState:
    topic_model: Optional[BERTopic] = None
    questions: List[str] = field(default_factory=list)
    topics: List[int] = field(default_factory=list)


state = AppState()
