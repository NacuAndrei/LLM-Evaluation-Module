import typing as t

from langchain_core.pydantic_v1 import BaseModel, Field
from ragas.llms.prompt import Prompt
from ragas.llms.output_parser import get_json_format_instructions


class Statements(BaseModel):
    sentence_index: int = Field(
        ...,
        description="Index of the sentence from the statement list. Always use 0 instead of null",  # This is the fix we needed to apply
    )
    simpler_statements: t.List[str] = Field(..., description="the simpler statements")


class StatementsAnswers(BaseModel):
    __root__: t.List[Statements]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_statements_output_instructions = get_json_format_instructions(StatementsAnswers)

CUSTOM_LONG_FORM_ANSWER_PROMPT = Prompt(
    name="long_form_answer",
    output_format_instruction=_statements_output_instructions,
    instruction="Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON.",
    examples=[
        {
            "question": "Who was Albert Einstein and what is he best known for?",
            "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            "sentences": """
        0:He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time.
        1:He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
        """,
            "analysis": StatementsAnswers.parse_obj(
                [
                    {
                        "sentence_index": 0,
                        "simpler_statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                        ],
                    },
                    {
                        "sentence_index": 1,
                        "simpler_statements": [
                            "Albert Einstein was best known for developing the theory of relativity.",
                            "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
                        ],
                    },
                ]
            ).dicts(),
        }
    ],
    input_keys=["question", "answer", "sentences"],
    output_key="analysis",
    language="english",
)
