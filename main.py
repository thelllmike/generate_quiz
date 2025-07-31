import os
import re
import json
import logging
from typing import List, Optional, Dict

import openai
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# fallback for exceptions if structure differs
try:
    from openai.error import OpenAIError
except ImportError:
    OpenAIError = Exception  # type: ignore

# load .env (optional)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quiz_api")

# ==== Pydantic schemas ====
class QuizRequest(BaseModel):
    technology: str = Field(..., example="React")
    num_questions: int = Field(10, ge=1, le=20, description="How many questions to generate")
    include_answers: bool = Field(False, description="Whether to include the correct answer in the output")

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: Optional[str] = None  # only included if include_answers=True

class AnswerItem(BaseModel):
    technology: str
    question: str
    correct_answer: str
    user_answer: str

class EvaluateRequest(BaseModel):
    answers: List[AnswerItem]

class EvaluationSummaryEntry(BaseModel):
    correct: int
    total: int
    percentage: float
    tuple: List[int]  # [correct, total]

# ==== App init ====
app = FastAPI(title="Tech Quiz Generator (gpt-3.5-turbo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client: Optional[OpenAI] = None  # will be set on startup

@app.on_event("startup")
def startup_event():
    global client
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.error("OPENAI_API_KEY not set.")
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")
    client = OpenAI(api_key=key)
    logger.info("OpenAI client initialized.")

# ==== Utility ====
def extract_json_array(raw: str) -> List[dict]:
    """
    Extract and parse the first JSON array from model output, with mild cleaning.
    """
    match = re.search(r"(\[.*\])", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model response.")
    json_str = match.group(1)

    def clean(s: str) -> str:
        # naive fixes: single to double quotes if needed
        if "'" in s and '"' not in s:
            s = s.replace("'", '"')
        # remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        cleaned = clean(json_str)
        return json.loads(cleaned)  # may still raise

# ==== Endpoints ====
@app.get("/")
def root():
    return {"status": "ok", "message": "Quiz generator up. POST /quiz with payload."}

@app.get("/diagnose")
def diagnose():
    """
    Helps debug what openai module is loaded.
    """
    info = {
        "openai_module_file": getattr(openai, "__file__", "unknown"),
        "openai_version": getattr(openai, "__version__", "unknown"),
    }
    return {"status": "ok", "openai_info": info}

@app.post("/quiz", response_model=List[QuizQuestion])
async def generate_quiz(req: QuizRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized.")

    tech = req.technology.strip()
    if not tech:
        raise HTTPException(status_code=400, detail="technology must be non-empty.")

    system_msg = (
        "You are a professional software engineering quiz writer. "
        "Generate clear, unambiguous multiple-choice questions about the specified technology."
    )

    example_with_answer = """
Example format:
[
  {
    "question": "What is JSX in React?",
    "options": ["A syntax extension for JavaScript", "A database", "A CSS framework", "A build tool"],
    "answer": "A syntax extension for JavaScript"
  }
]
"""

    example_without_answer = """
Example format:
[
  {
    "question": "What is the JVM in Java?",
    "options": ["Java Virtual Machine", "Java Version Manager", "Just Virtual Memory", "Jump Value Marker"]
  }
]
"""

    user_instructions = (
        f"Generate {req.num_questions} multiple-choice questions about \"{tech}\". "
        f"Each question must have exactly 4 options. "
        + (
            "Include the correct answer in an \"answer\" field." if req.include_answers
            else "Do NOT include any answer field; only questions and options."
        )
        + " Respond with a single valid JSON array and nothing else. "
        + (example_with_answer if req.include_answers else example_without_answer)
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_instructions},
            ],
            temperature=0.6,
            max_tokens=1200,
        )
    except OpenAIError as e:
        logger.exception("OpenAI API error")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e}")
    except Exception as e:
        logger.exception("Unexpected error calling OpenAI")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    # Access content
    try:
        raw_output = resp.choices[0].message.content  # typical structure
    except Exception:
        raw_output = getattr(resp.choices[0].message, "content", "") if resp.choices else ""
    logger.debug("Raw model output: %s", raw_output[:1000])

    # Parse
    try:
        items = extract_json_array(raw_output)
    except Exception as e:
        logger.error("Failed to parse JSON: %s", e)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to parse JSON from model response: {e}. "
                f"Raw output snippet: {raw_output[:1000]!r}"
            ),
        )

    output: List[Dict] = []
    for obj in items:
        question = obj.get("question") or obj.get("prompt")
        options = obj.get("options")
        if not question or not isinstance(options, list) or len(options) != 4:
            continue
        answer = obj.get("answer") if req.include_answers else None
        output.append({
            "question": question.strip(),
            "options": [str(o).strip() for o in options],
            "answer": answer.strip() if isinstance(answer, str) else None,
        })

    if not output:
        raise HTTPException(status_code=500, detail="No valid questions extracted from model output.")

    return output

@app.post("/evaluate")
async def evaluate(req: EvaluateRequest, simple: bool = Query(False, description="If true, return only tuples [correct, total] per technology")):
    """
    Input: list of answered items with technology, correct_answer, and user_answer.
    Output: per-technology summary of correct/total and percentage.
    """
    if not req.answers:
        raise HTTPException(status_code=400, detail="No answers provided.")

    summary: Dict[str, Dict[str, int]] = {}

    for item in req.answers:
        tech = item.technology.strip()
        is_correct = item.user_answer.strip().lower() == item.correct_answer.strip().lower()
        entry = summary.setdefault(tech, {"correct": 0, "total": 0})
        if is_correct:
            entry["correct"] += 1
        entry["total"] += 1

    if simple:
        # minimal tuple-only
        return {tech: [v["correct"], v["total"]] for tech, v in summary.items()}

    result: Dict[str, EvaluationSummaryEntry] = {}
    for tech, v in summary.items():
        correct = v["correct"]
        total = v["total"]
        percentage = round(100 * correct / total, 1) if total > 0 else 0.0
        entry = EvaluationSummaryEntry(
            correct=correct,
            total=total,
            percentage=percentage,
            tuple=[correct, total],
        )
        result[tech] = entry

    return result
