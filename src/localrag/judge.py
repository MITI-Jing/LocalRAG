""""LLM-as-judge: gemma2:9b grades each generated answer against the answer key.

Two cases never reach the LLM and are graded by rule (see grade()). The rules, 
JUDGE_PROMPT and the JudgeScore field descriptions together are one judge version:
change any of them and bump JUDGE_PROMPT_V.

"""

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from localrag.pipeline import is_abstention

JUDGE_MODEL = "gemma2:9b"
JUDGE_PROMPT_V = "v2"


class JudgeScore(BaseModel):
    """Scores for one RAG answer, each 0.0-1.0."""
    faithfulness: float = Field(
        ge=0, le=1,
        description="Is every claim in the answer supported by the retrieved context?"
        "1.0 = fully grounded, 0.0 = hallucinated.")
    correctness: float = Field(
        ge=0, le=1,
        description="Does the answer match the ground-truth answer key?"
                    "1.0 = same facts, 0.0 = wrong.")
    completeness: float = Field(
        ge=0, le=1,
        description="Does it include ALL facts/items from the answer key?"
                    "Lower if it dropped items from a list.")
    reason: str = Field(description="One sentence justifying the scores.")

judge = ChatOllama(model=JUDGE_MODEL,temperature=0, num_predict=512, num_ctx=8192)
judge_structured = judge.with_structured_output(JudgeScore)



JUDGE_PROMPT = (
    "You are grading a RAG system's answer.\n\n"

    "QUESTION: {question}\n\n"

    "GROUND-TRUTH ANSWER(the answer key) : {gold}\n\n"

    "RETRIEVED CONTEXT the system saw:{context}\n\n"

    "SYSTEM'S ANSWER: {response}\n\n"

    "Score faithfulness, correctness, and completeness.\n"
    "Correctness and completeness are graded ONLY against the answer key.\n"
    "The answer may be worded differently from the answer key; grade semantic equivalence,"
    " not exact wording.\n"
    "Extra information beyond the answer key must NOT lower any score,"
    " as long as it supported by the context. \n"
    "The context is used only for faithfulness.\n"
    "IMPORTANT: each score MUST be a decimal number between 0.0 and 1.0\n"
    "(for example 0.0, 0.25, 0.5, 0.75, 1.0). Do NOT use a 1-5 or 1-10 scale.\n"
    "1.0 means perfect, 0.0 means total failure.\n"
    "Then give a one-sentence reason."
)

def judge_answer(question, gold, response, context) -> JudgeScore:
    return judge_structured.invoke(JUDGE_PROMPT.format(
        question=question, gold=gold, response=response, context=context))

def grade(qtype: str, question: str, gold: str, response: str, context: str ) -> JudgeScore:
    """The whole grading policy: abstentions are scored by rule, everything else by the LLM."""
    abstained = is_abstention(response)
    if qtype == "no_answer":
        # gold is NOT_IN_DOCUMENT, which no judge would match against "Not found in context."
        return JudgeScore(faithfulness=1.0, correctness=1.0 if abstained else 0.1,
                          completeness=1.0, reason="deterministic abstention check")
    if abstained:
        return JudgeScore(faithfulness=1.0, correctness=0.0, completeness=0.0,
                          reason="deterministic:abstained on answerable question")
    return judge_answer(question, gold, response, context)

