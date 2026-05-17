"""
Examples:
  # OpenAI-compatible (default)
  export OPENAI_API_KEY="your_key_here"
  python launch_over_chexbench.py --max-cases 2 --model chatgpt-4o-latest

  # Gemini via OpenAI-compatible proxy (tools disabled for compatibility)
  export OPENAI_API_KEY=xx
  export OPENAI_BASE_URL="https://your-openai-compatible-gemini-endpoint/v1"
  python launch_over_chexbench.py --max-cases 2 --model gemini-1.5-pro

  # Gemini native (tools supported)
  export GEMINI_API_KEY="your_key_here"
  python launch_over_chexbench.py --max-cases 2 --model gemini-3-flash-preview --gemini-native

  export GEMINI_API_KEY="your_key_here"
  python launch_over_chexbench.py --data-file data/chestagentbench/metadata.jsonl --llm-parse --model gemini-3-flash-preview
  python launch_over_chexbench.py --data-file data/mimic/medrax_input_all_2000.jsonl --llm-parse --model gemini-3-flash-preview --log-prefix agent-gemini3flash-mimic
"""
import argparse
import json
import logging
import os
import re
import signal
import base64
from glob import glob
from datetime import datetime
from typing import List, Optional, Set, Tuple

from datasets import load_dataset
import openai
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("chexbench_agent")


def setup_logging(filename: str) -> logging.Logger:
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tool-using CXR agent over the ChestAgentBench dataset."
    )
    parser.add_argument("--prompt-file", default="medrax/docs/system_prompts.txt")
    parser.add_argument("--tools", default="")
    parser.add_argument("--model-dir", default=os.getenv("MEDRAX_MODEL_DIR", "model-weights"))
    parser.add_argument("--temp-dir", default=os.getenv("MEDRAX_TEMP_DIR", "temp"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="chatgpt-4o-latest")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--log-prefix", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--data-file", default="data/chestagentbench/metadata.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--answer-retries", type=int, default=2)
    parser.add_argument("--llm-parse", action="store_true")
    parser.add_argument("--disable-tools", action="store_true")
    parser.add_argument("--disable-parallel-tool-calls", action="store_true")
    parser.add_argument("--gemini-native", action="store_true")
    parser.add_argument("--gemini-api-key", type=str, default=None)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a previous run log file or log directory for resuming unfinished questions.",
    )
    parser.add_argument(
        "--resume-statuses",
        type=str,
        default="ok,skipped,invalid_answer",
        help="Comma-separated statuses to treat as completed when resuming.",
    )
    return parser.parse_args()


def parse_tools(tools_csv: str) -> Optional[List[str]]:
    tools_csv = tools_csv.strip()
    if not tools_csv:
        return None
    return [tool.strip() for tool in tools_csv.split(",") if tool.strip()]


def parse_statuses(statuses_csv: str) -> Set[str]:
    return {status.strip() for status in (statuses_csv or "").split(",") if status.strip()}


def resolve_resume_files(resume_from: str, log_prefix: str) -> List[str]:
    if not resume_from:
        return []
    if os.path.isfile(resume_from):
        return [resume_from]
    if not os.path.isdir(resume_from):
        raise SystemExit(f"--resume-from path does not exist: {resume_from}")

    candidates = sorted(
        glob(os.path.join(resume_from, f"{log_prefix}_*.json"))
        + glob(os.path.join(resume_from, f"{log_prefix}_*.jsonl"))
    )
    if not candidates:
        candidates = sorted(
            [
                path
                for path in glob(os.path.join(resume_from, "*.json"))
                + glob(os.path.join(resume_from, "*.jsonl"))
                if not os.path.basename(path).startswith("tool_calls_")
            ]
        )
    return candidates


def load_completed_question_ids(
    log_files: List[str], completed_statuses: Set[str]
) -> Tuple[Set[str], int, dict]:
    completed = set()
    parsed_lines = 0
    status_by_qid = {}
    correct_by_qid = {}
    for log_file in log_files:
        try:
            with open(log_file, "r") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parsed_lines += 1
                    if not isinstance(entry, dict):
                        continue
                    question_id = entry.get("question_id")
                    status = entry.get("status")
                    if question_id is None or status not in completed_statuses:
                        continue
                    qid = str(question_id)
                    completed.add(qid)
                    status_by_qid[qid] = status
                    correct_by_qid[qid] = bool(entry.get("is_correct", False))
        except OSError as exc:
            print(f"Warning: could not read resume log {log_file}: {exc}")
    ok_count = sum(1 for s in status_by_qid.values() if s == "ok")
    skipped_like_count = len(status_by_qid) - ok_count
    correct_count = sum(
        1
        for qid, status in status_by_qid.items()
        if status == "ok" and correct_by_qid.get(qid, False)
    )
    summary = {
        "completed": len(status_by_qid),
        "ok": ok_count,
        "skipped_like": skipped_like_count,
        "correct": correct_count,
    }
    return completed, parsed_lines, summary


def normalize_image_paths(raw_paths) -> List[str]:
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]
    elif isinstance(raw_paths, list) and raw_paths and isinstance(raw_paths[0], list):
        raw_paths = [path for sublist in raw_paths for path in sublist]

    normalized = []
    for path in raw_paths or []:
        if not path or not isinstance(path, str):
            continue
        clean_path = path.replace("figures/", "")
        full_path = os.path.join("figures", clean_path)
        if os.path.exists(full_path):
            normalized.append(full_path)
    return normalized


def encode_image(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        return None


def build_multimodal_content(text: str, image_paths: List[str], extra_text: Optional[str] = None):
    content = [{"type": "text", "text": text}]
    if extra_text:
        content.append({"type": "text", "text": extra_text})
    for path in image_paths:
        encoded = encode_image(path)
        if not encoded:
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return content


def serialize_messages(messages):
    serialized = []
    for msg in messages or []:
        try:
            entry = {
                "type": getattr(msg, "type", msg.__class__.__name__),
                "role": getattr(msg, "role", None),
                "name": getattr(msg, "name", None),
                "content": getattr(msg, "content", None),
                "tool_calls": getattr(msg, "tool_calls", None),
                "additional_kwargs": getattr(msg, "additional_kwargs", None),
            }
            serialized.append(entry)
        except Exception:
            serialized.append({"type": str(type(msg)), "content": str(msg)})
    return serialized


def build_message(example: dict, image_paths: List[str]) -> str:
    question = example.get("question", "")
    explanation = example.get("explanation", "")
    lines = [
        "Given the following medical case:",
        question,
        "",
        "Base your answer only on the provided images and case information.",
    ]
    if explanation:
        lines.extend(["", "Case details:", explanation])
    lines.append("")
    lines.append("Image paths (local files):")
    for path in image_paths:
        lines.append(f"- {path}")
    lines.append("")
    lines.append("Use available tools as needed. When calling tools, use the image paths exactly.")
    return "\n".join(lines)


def extract_choice(text: str) -> Optional[str]:
    if text is None:
        return None
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    elif isinstance(text, dict):
        text = json.dumps(text)
    else:
        text = str(text)
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.fullmatch(r"([A-F])", line, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    matches = re.findall(r"\b([A-F])\b", text, re.IGNORECASE)
    return matches[-1].upper() if matches else None


def parse_choice_with_llm(client, model: str, content: Optional[str]) -> Optional[str]:
    if not content:
        return None
    messages = [
        {"role": "system", "content": "Extract the single answer choice letter (A-F) from the model output."},
        {"role": "user", "content": content},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=5,
            temperature=0.0,
        )
        parsed = response.choices[0].message.content if response.choices else None
        return extract_choice(parsed or "")
    except Exception:
        return None


def parse_choice_with_gemini(model, content: Optional[str]) -> Optional[str]:
    if not content:
        return None
    messages = [
        SystemMessage(content="Extract the single answer choice letter (A-F) from the model output."),
        HumanMessage(content=content),
    ]
    try:
        response = model.invoke(messages)
        return extract_choice(getattr(response, "content", "") or "")
    except Exception:
        return None


def answer_with_llm(client, model: str, prompt_text: str) -> Optional[str]:
    if not prompt_text:
        return None
    messages = [
        {"role": "system", "content": "You are a medical imaging expert. Reply with a single letter A-F only."},
        {"role": "user", "content": f"{prompt_text}\n\nAnswer with a single letter A-F only."},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=5,
            temperature=0.0,
        )
        content = response.choices[0].message.content if response.choices else None
        return extract_choice(content or "")
    except Exception:
        return None


def answer_with_gemini(model, prompt_text: str) -> Optional[str]:
    if not prompt_text:
        return None
    messages = [
        SystemMessage(content="You are a medical imaging expert. Reply with a single letter A-F only."),
        HumanMessage(content=f"{prompt_text}\n\nAnswer with a single letter A-F only."),
    ]
    try:
        response = model.invoke(messages)
        return extract_choice(getattr(response, "content", "") or "")
    except Exception:
        return None


def invoke_with_retries(
    agent,
    base_messages: List[dict],
    max_retries: int,
    llm_parse: bool,
    parse_choice_fn,
    answer_fn,
    thread_id: str,
):
    response_text = None
    predicted = None
    trace = []
    attempts = 0
    while attempts <= max_retries:
        attempts += 1
        attempt_messages = base_messages
        if attempts > 1:
            attempt_messages = base_messages + [
                {"role": "user", "content": "Answer with a single letter A-F only. No other text."}
            ]
        result = agent.workflow.invoke(
            {"messages": attempt_messages},
            config={"configurable": {"thread_id": thread_id}},
        )
        trace = serialize_messages(result.get("messages"))
        response_text = result["messages"][-1].content if result.get("messages") else None
        predicted = extract_choice(response_text or "")
        if not predicted and llm_parse and parse_choice_fn:
            predicted = parse_choice_fn(response_text)
        if predicted:
            return response_text, predicted, attempts, trace
    if llm_parse and base_messages and answer_fn:
        prompt_text = base_messages[0].get("content", "")
        fallback = answer_fn(prompt_text)
        if fallback:
            return response_text, fallback, attempts + 1, trace
    return response_text, predicted, attempts, trace


def main() -> None:
    args = parse_args()
    is_gemini_model = args.model.lower().startswith("gemini-")
    if is_gemini_model and not args.gemini_native and not args.disable_tools:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            args.gemini_native = True
            print("Detected GEMINI_API_KEY/GOOGLE_API_KEY with a Gemini model; enabling --gemini-native for tool calls.")
        else:
            base_url = os.getenv("OPENAI_BASE_URL", "")
            if "generativelanguage.googleapis.com" in base_url:
                raise SystemExit(
                    "Gemini OpenAI-compatible endpoints do not support tool calls. "
                    "Set GEMINI_API_KEY (or GOOGLE_API_KEY) and add --gemini-native to enable tools."
                )
    tools = parse_tools(args.tools)
    if tools is None:
        # Default: exclude generator tool to avoid requiring RoentGen weights.
        tools = [
            "ImageVisualizerTool",
            "DicomProcessorTool",
            "ChestXRayClassifierTool",
            "ChestXRaySegmentationTool",
            "ChestXRayReportGeneratorTool",
            "XRayVQATool",
            "LlavaMedTool",
            "XRayPhraseGroundingTool",
        ]

    config = {
        "prompt_file": args.prompt_file,
        "tools": tools,
        "model_dir": args.model_dir,
        "temp_dir": args.temp_dir,
        "device": args.device,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_cases": args.max_cases,
    }

    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": config}, indent=2))
        return

    openai_kwargs = {}
    parse_client = None
    parse_choice_fn = None
    answer_fn = None
    direct_model = None
    gemini_key = None

    if args.gemini_native:
        gemini_key = (
            args.gemini_api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not gemini_key:
            raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        if args.disable_tools:
            tools = []
        use_direct = len(tools) == 0
        if use_direct or args.llm_parse:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except Exception as exc:
                raise SystemExit(
                    "Gemini native backend requires `langchain-google-genai`. "
                    "Install it with: pip install langchain-google-genai google-generativeai"
                ) from exc
            if use_direct:
                direct_model = ChatGoogleGenerativeAI(
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    google_api_key=gemini_key,
                )
            if args.llm_parse:
                parser_model = ChatGoogleGenerativeAI(
                    model=args.model,
                    temperature=0.0,
                    top_p=1.0,
                    google_api_key=gemini_key,
                )
                parse_choice_fn = lambda content: parse_choice_with_gemini(parser_model, content)
                answer_fn = lambda prompt: answer_with_gemini(parser_model, prompt)
    else:
        from medrax.utils.utils import resolve_openai_client_kwargs

        openai_kwargs = resolve_openai_client_kwargs()
        if "api_key" not in openai_kwargs:
            raise SystemExit("OPENAI_API_KEY environment variable is not set.")

        base_url = openai_kwargs.get("base_url", "")
        if base_url:
            if "generativelanguage.googleapis.com" in base_url and args.tools.strip() == "" and not args.disable_tools:
                print("Warning: Gemini OpenAI-compatible endpoints do not support tool calls without thought signatures. Disabling tools.")
                tools = []
        if args.disable_tools:
            tools = []
        use_direct = len(tools) == 0
        parse_client = openai.OpenAI(**openai_kwargs)
        if args.llm_parse:
            parse_choice_fn = lambda content: parse_choice_with_llm(parse_client, args.model, content)
            answer_fn = lambda prompt: answer_with_llm(parse_client, args.model, prompt)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", args.model_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.model_dir)
    os.environ.setdefault("TORCH_HOME", args.model_dir)

    log_prefix = args.log_prefix or f"agent_{args.model}"
    log_dir = os.path.join("logs", log_prefix)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    setup_logging(log_filename)

    shutdown = {"requested": False}

    def signal_handler(signum, frame):
        shutdown["requested"] = True
        print("\nShutdown signal received. Completing current task...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    dataset = load_dataset("json", data_files=args.data_file)
    train_dataset = dataset["train"]

    if args.max_cases is not None:
        train_dataset = train_dataset.select(range(min(args.max_cases, len(train_dataset))))

    completed_from_resume = 0
    resume_summary = {"completed": 0, "ok": 0, "skipped_like": 0, "correct": 0}
    if args.resume_from:
        completed_statuses = parse_statuses(args.resume_statuses)
        resume_files = resolve_resume_files(args.resume_from, log_prefix)
        completed_ids, parsed_lines, resume_summary = load_completed_question_ids(
            resume_files, completed_statuses
        )
        if resume_files:
            print(
                f"Resume scan: parsed {parsed_lines} log lines from {len(resume_files)} file(s), "
                f"found {len(completed_ids)} completed question IDs for statuses {sorted(completed_statuses)}."
            )
        else:
            print(f"Resume scan: no log files found under {args.resume_from}; starting from scratch.")
        if completed_ids:
            keep_indices = []
            for idx, example in enumerate(train_dataset):
                question_id = str(example.get("question_id", "unknown"))
                if question_id not in completed_ids:
                    keep_indices.append(idx)
            completed_from_resume = len(train_dataset) - len(keep_indices)
            train_dataset = train_dataset.select(keep_indices)

    agent = None
    parallel_tool_calls = None
    if use_direct:
        print("Tools disabled: using direct model calls for evaluation.")
    else:
        if args.disable_parallel_tool_calls:
            parallel_tool_calls = False
        from main import initialize_agent

        agent, _ = initialize_agent(
            args.prompt_file,
            tools_to_use=tools,
            model_dir=args.model_dir,
            temp_dir=args.temp_dir,
            device=args.device,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            openai_kwargs=openai_kwargs,
            llm_backend="gemini" if args.gemini_native else "openai",
            llm_api_key=gemini_key,
            log_dir=log_dir,
            parallel_tool_calls=parallel_tool_calls,
        )

    total = len(train_dataset)
    processed = 0
    skipped = 0
    correct = 0

    print(f"Beginning agent evaluation for model {args.model}")
    print(f"Temperature: {args.temperature}")
    if completed_from_resume:
        print(f"Resumed mode: skipped {completed_from_resume} previously completed cases.")
    print(f"Processing {total} cases")

    for example in train_dataset:
        if shutdown["requested"]:
            print("\nGraceful shutdown initiated. Saving progress...")
            break

        processed += 1
        image_paths = normalize_image_paths(example.get("images"))
        if not image_paths:
            skipped += 1
            log_entry = {
                "question_id": example.get("question_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "skipped",
                "reason": "no_images",
                "input": {
                    "question": example.get("question"),
                    "explanation": example.get("explanation", ""),
                    "images": example.get("images"),
                },
            }
            logger.info(json.dumps(log_entry))
            print(f"Skipped question: {example.get('question_id', 'unknown')}")
            continue

        message = build_message(example, image_paths)
        messages = [{"role": "user", "content": message}]

        try:
            if use_direct:
                attempts = 0
                response_text = None
                predicted = None
                trace = []
                while attempts <= max(0, args.answer_retries):
                    attempts += 1
                    extra = None
                    if attempts > 1:
                        extra = "Answer with a single letter A-F only. No other text."
                    content = build_multimodal_content(message, image_paths, extra_text=extra)
                    if args.gemini_native:
                        system_msg = SystemMessage(
                            content="You are a medical imaging expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."
                        )
                        user_msg = HumanMessage(content=content)
                        response = direct_model.invoke([system_msg, user_msg])
                        response_text = getattr(response, "content", None)
                    else:
                        response = parse_client.chat.completions.create(
                            model=args.model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a medical imaging expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F).",
                                },
                                {"role": "user", "content": content},
                            ],
                            max_tokens=50,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                        response_text = response.choices[0].message.content if response.choices else None
                    predicted = extract_choice(response_text or "")
                    if not predicted and args.llm_parse and parse_choice_fn:
                        predicted = parse_choice_fn(response_text)
                    if predicted:
                        break
                if not predicted and args.llm_parse and answer_fn:
                    predicted = answer_fn(messages[0]["content"])
            else:
                response_text, predicted, attempts, trace = invoke_with_retries(
                    agent,
                    messages,
                    max(0, args.answer_retries),
                    args.llm_parse,
                    parse_choice_fn,
                    answer_fn,
                    thread_id=str(example.get("question_id", processed)),
                )
            if not predicted:
                skipped += 1
                log_entry = {
                    "question_id": example.get("question_id", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                    "status": "invalid_answer",
                    "model": args.model,
                    "temperature": args.temperature,
                    "attempts": attempts,
                    "input": {
                        "question": example.get("question"),
                        "explanation": example.get("explanation", ""),
                        "images": image_paths,
                    },
                    "model_answer": response_text,
                    "predicted_answer": predicted,
                    "correct_answer": example.get("answer"),
                    "trace": trace,
                }
                logger.info(json.dumps(log_entry))
                print(f"Skipped question: {example.get('question_id', 'unknown')} (invalid answer)")
                continue
            expected = example.get("answer")
            is_correct = bool(predicted and expected and predicted == expected)
            if is_correct:
                correct += 1
            log_entry = {
                "question_id": example.get("question_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "ok",
                "model": args.model,
                "temperature": args.temperature,
                "input": {
                    "question": example.get("question"),
                    "explanation": example.get("explanation", ""),
                    "images": image_paths,
                },
                "model_answer": response_text,
                "predicted_answer": predicted,
                "correct_answer": expected,
                "is_correct": is_correct,
                "attempts": attempts,
                "trace": trace,
            }
            logger.info(json.dumps(log_entry))
            print(f"Progress: {processed}/{total}")
            print(f"Question ID: {example.get('question_id', 'unknown')}")
            print(f"Model Answer: {response_text}")
            print(f"Parsed Answer: {predicted}")
            print(f"Correct Answer: {expected}\n")
        except Exception as exc:
            log_entry = {
                "question_id": example.get("question_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(exc),
                "input": {
                    "question": example.get("question"),
                    "explanation": example.get("explanation", ""),
                    "images": image_paths,
                },
            }
            logger.info(json.dumps(log_entry))
            print(f"Error processing question {example.get('question_id', 'unknown')}: {exc}")
            raise

    print("\nBenchmark Summary:")
    total_processed_all = processed + resume_summary["completed"]
    total_skipped_all = skipped + resume_summary["skipped_like"]
    total_correct_all = correct + resume_summary["correct"]
    print(f"Total Examples Processed: {total_processed_all}")
    if completed_from_resume:
        print(f"Previously Completed (Skipped by Resume): {completed_from_resume}")
    print(f"Total Examples Skipped: {total_skipped_all}")
    if total_processed_all - total_skipped_all > 0:
        accuracy = total_correct_all / max(1, total_processed_all - total_skipped_all)
        print(
            f"Accuracy: {accuracy:.4f} "
            f"({total_correct_all}/{total_processed_all - total_skipped_all})"
        )
    if os.path.exists(log_filename) and os.path.getsize(log_filename) > 0:
        print(f"\nLog file saved to: {os.path.abspath(log_filename)}")
    else:
        print(f"\nWarning: Log file could not be verified at: {os.path.abspath(log_filename)}")


if __name__ == "__main__":
    main()
