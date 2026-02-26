from typing import Optional, List
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime


def get_latest_log() -> str:
    """Find the most recently modified log file in the current directory.

    Returns:
        str: Path to the most recently modified log file

    Raises:
        FileNotFoundError: If no log files are found in the current directory
    """
    logs = list(Path(".").glob("api_usage_*.json"))
    if not logs:
        raise FileNotFoundError("No log files found in the current directory.")
    return str(max(logs, key=lambda p: p.stat().st_mtime))


def format_cost(entry: dict) -> str:
    """Format cost if available, otherwise return 'N/A'

    Args:
        entry: Log entry dictionary containing cost information

    Returns:
        str: Formatted cost string with $ and 4 decimal places, or 'N/A' if cost not found
    """
    return f"${entry.get('cost', 'N/A'):.4f}" if "cost" in entry else "N/A"


def print_gpt4_entry(entry: dict) -> None:
    """Print entry for GPT-4 format

    Args:
        entry: Log entry dictionary in GPT-4 format containing model info, inputs and outputs
    """
    print("\n=== Log Entry ===")
    print(f"Model: {entry['model']}")
    print(f"Case ID: {entry['case_id']}")
    print(f"Question ID: {entry['question_id']}")

    print("\n=== Model Input ===")
    messages = entry["input"]["messages"]
    print("System message:", messages[0]["content"])
    user_content = messages[1]["content"]
    print("\nUser prompt:", user_content[0]["text"])
    print("\nImages provided:")
    for content in user_content[1:]:
        print(f"  - {content['image_url']['url']}")

    print("\n=== Model Output ===")
    print(f"Answer: {entry['model_answer']}")
    print(f"Correct: {entry['correct_answer']}")

    print("\n=== Usage Stats ===")
    print(f"Duration: {entry['duration']}s")
    print(f"Cost: {format_cost(entry)}")
    print(
        f"Tokens: {entry['usage']['total_tokens']}",
        f"(prompt: {entry['usage']['prompt_tokens']},",
        f"completion: {entry['usage']['completion_tokens']})",
    )


def print_llama_entry(entry: dict) -> None:
    """Print entry for Llama-3.2 format

    Args:
        entry: Log entry dictionary in Llama format containing model info, inputs and outputs
    """
    print("\n=== Log Entry ===")
    print(f"Model: {entry['model']}")
    print(f"Case ID: {entry['case_id']}")
    print(f"Question ID: {entry['question_id']}")

    print("\n=== Model Input ===")
    print(f"Question: {entry['input']['question_data']['question']}")
    print("\nImages provided:")
    for url in entry["input"]["image_urls"]:
        print(f"  - {url}")
    if entry["input"]["image_captions"]:
        print("\nImage captions:")
        for caption in entry["input"]["image_captions"]:
            if caption:
                print(f"  - {caption}")

    print("\n=== Model Output ===")
    print(f"Answer: {entry['model_answer']}")
    print(f"Correct: {entry['correct_answer']}")

    print("\n=== Usage Stats ===")
    print(f"Duration: {entry['duration']}s")
    if "usage" in entry:
        print(
            f"Tokens: {entry['usage']['total_tokens']}",
            f"(prompt: {entry['usage']['prompt_tokens']},",
            f"completion: {entry['usage']['completion_tokens']})",
        )


def determine_model_type(entry: dict) -> str:
    """Determine the model type from the entry

    Args:
        entry: Log entry dictionary containing model information

    Returns:
        str: Model type - 'gpt4', 'llama', or 'unknown'
    """
    model = entry.get("model", "").lower()
    if "gpt-4" in model:
        return "gpt4"
    elif "llama" in model:
        return "llama"
    elif "chexagent" in model:
        return "chexagent"
    elif "medrax" in model:
        return "medrax"
    else:
        return "unknown"


def print_log_entry(
    log_file: Optional[str] = None,
    num_entries: Optional[int] = None,
    model_filter: Optional[str] = None,
) -> None:
    """Print log entries from the specified log file or the latest log file.

    Args:
        log_file: Path to the log file. If None, uses the latest log file.
        num_entries: Number of entries to print. If None, prints all entries.
        model_filter: Filter entries by model type ('gpt4' or 'llama'). If None, prints all.
    """
    if log_file is None:
        log_file = get_latest_log()
        print(f"Using latest log file: {log_file}")

    entries_printed = 0
    total_entries = 0
    filtered_entries = 0

    with open(log_file, "r") as f:
        for line in f:
            if line.startswith("HTTP"):
                continue
            try:
                total_entries += 1
                entry = json.loads(line)

                # Apply model filter if specified
                model_type = determine_model_type(entry)
                if model_filter and model_type != model_filter:
                    filtered_entries += 1
                    continue

                if model_type == "gpt4":
                    print_gpt4_entry(entry)
                elif model_type == "llama":
                    print_llama_entry(entry)
                else:
                    print(f"Unknown model type in entry: {entry['model']}")
                    continue

                print("=" * 50)
                entries_printed += 1
                if num_entries and entries_printed >= num_entries:
                    break

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing entry: {e}")
                continue

    print(f"\nSummary:")
    print(f"Total entries: {total_entries}")
    print(f"Entries printed: {entries_printed}")
    if model_filter:
        print(f"Entries filtered: {filtered_entries}")


def main() -> None:
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Parse and display log entries from API usage logs."
    )
    parser.add_argument("-l", "--log_file", nargs="?", help="Path to the log file (optional)")
    parser.add_argument("-n", "--num_entries", type=int, help="Number of entries to display")
    parser.add_argument(
        "-m",
        "--model",
        choices=["gpt4", "llama"],
        default="gpt4",
        help="Model type to display (default: gpt4)",
    )
    args = parser.parse_args()

    try:
        print_log_entry(args.log_file, args.num_entries, args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
