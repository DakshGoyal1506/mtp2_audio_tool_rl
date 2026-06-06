import pydantic
from typing import List


class SFT_GRPO(pydantic.BaseModel):
    id: str                 # Audio file ID (relative audio filepath)
    reasoning: str          # Model's step-by-step reasoning/thinking process
    question: str           # Hard MMAU-style question (no direct tool mentions)
    options: List[str]      # 4 options to choose from
    correct_answer: str     # The correct option (should exactly match one of the options)
    tool: str               # The tool this question was generated for


class SFT_GRPO_Dataset(pydantic.BaseModel):
    data: List[SFT_GRPO]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> SFT_GRPO:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __str__(self) -> str:
        return f"SFT_GRPO_Dataset with {len(self.data)} samples"
