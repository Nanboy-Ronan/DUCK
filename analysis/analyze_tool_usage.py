#!/usr/bin/env python3
"""
Analyze agent run performance with a focus on tool usage from a log file.

This script parses a log file (either JSON or JSONL), extracts tool call information,
and generates a set of summary artifacts. It is designed to be robust to variations
in log schema and file format.

Usage Examples:
    # Analyze the default log file and write to the default output directory
    python analysis/analyze_tool_usage.py

    # Analyze a specific log file and write to a custom output directory
    python analysis/analyze_tool_usage.py \
        --input /path/to/another/log.json \
        --outdir /path/to/custom/output

    # Analyze a JSONL file
    python analysis/analyze_tool_usage.py \
        --input /path/to/log.jsonl
"""
import argparse
import json
import os
import pandas as pd
from collections import defaultdict
import logging
from glob import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Heuristics for finding relevant fields
TOOL_NAME_KEYS = ['tool', 'tool_name', 'name', 'function', 'action']
TOOL_CALL_KEYS = ['tool_call', 'tool_calls', 'tools', 'function_call', 'function_calls']
TOOL_RESULT_KEYS = ['result', 'response', 'output', 'error', 'status']
LATENCY_KEYS = ['latency_ms', 'duration_ms', 'elapsed_ms', 'time_ms']
SESSION_ID_KEYS = ['run_id', 'request_id', 'conversation_id', 'turn_id', 'session_id']

def detect_log_format(file_path):
    """
    Detect if the file is JSON or JSONL.
    """
    with open(file_path, 'r') as f:
        first_char = f.read(1)
        if first_char == '[':
            return 'json'
        return 'jsonl'

def parse_log_file(file_path):
    """
    Parse a log file, supporting both JSON and JSONL formats.
    """
    log_format = detect_log_format(file_path)
    logging.info(f"Detected log format: {log_format.upper()}")

    with open(file_path, 'r') as f:
        if log_format == 'json':
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data, log_format
                else:
                    return [data], log_format
            except json.JSONDecodeError:
                logging.error("Failed to parse JSON file.")
                return [], log_format
        else:  # JSONL
            return [json.loads(line) for line in f if line.strip()], log_format

def find_nested_key(data, keys):
    """
    Find the first matching key in a nested dictionary.
    """
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        for key in data:
            result = find_nested_key(data[key], keys)
            if result is not None:
                return result
    return None

def extract_tool_calls(events):
    """
    Extract tool call information from a list of log events.
    """
    tool_calls = []
    for event in events:
        session_id = find_nested_key(event, SESSION_ID_KEYS) or event.get('question_id', 'default_session')
        
        trace = event.get('trace', [])
        if trace:
            for item in trace:
                potential_tool_calls = find_nested_key(item, TOOL_CALL_KEYS)
                if not potential_tool_calls:
                    if any(key in item for key in TOOL_NAME_KEYS):
                        potential_tool_calls = [item]
                    else:
                        continue
                
                if not isinstance(potential_tool_calls, list):
                    potential_tool_calls = [potential_tool_calls]

                for call in potential_tool_calls:
                    if not isinstance(call, dict):
                        continue

                    tool_name = find_nested_key(call, TOOL_NAME_KEYS)
                    if not tool_name:
                        continue

                    result = find_nested_key(call, TOOL_RESULT_KEYS)
                    latency = find_nested_key(call, LATENCY_KEYS)
                    status = 'success' if find_nested_key(call, ['error']) is None else 'failure'
                    
                    if 'status' in call:
                        status = call['status']

                    error_message = find_nested_key(call, ['error', 'error_message'])
                    
                    tool_calls.append({
                        'session_id': session_id,
                        'tool_name': tool_name,
                        'status': status,
                        'latency_ms': latency,
                        'result': result,
                        'error_message': error_message,
                    })
        else:
            potential_tool_calls = find_nested_key(event, TOOL_CALL_KEYS)
            if not potential_tool_calls:
                if any(key in event for key in TOOL_NAME_KEYS):
                     potential_tool_calls = [event]
                else:
                    continue

            if not isinstance(potential_tool_calls, list):
                potential_tool_calls = [potential_tool_calls]
            
            for call in potential_tool_calls:
                if not isinstance(call, dict):
                    continue
                
                tool_name = find_nested_key(call, TOOL_NAME_KEYS)
                if not tool_name:
                    continue

                result = find_nested_key(call, TOOL_RESULT_KEYS)
                latency = find_nested_key(call, LATENCY_KEYS)
                status = 'success' if find_nested_key(call, ['error']) is None else 'failure'
                
                if 'status' in call:
                    status = call['status']

                error_message = find_nested_key(call, ['error', 'error_message'])
                
                tool_calls.append({
                    'session_id': session_id,
                    'tool_name': tool_name,
                    'status': status,
                    'latency_ms': latency,
                    'result': result,
                    'error_message': error_message,
                })
    return tool_calls

def analyze_tool_usage(tool_calls):
    """
    Analyze the extracted tool calls and generate summary statistics.
    """
    if not tool_calls:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(tool_calls)

    # Overall summary
    summary = {
        'total_tool_calls': len(df),
        'successful_calls': int(df['status'].apply(lambda x: str(x).lower() == 'success').sum()),
        'failed_calls': int(df['status'].apply(lambda x: str(x).lower() != 'success').sum()),
        'average_latency_ms': df['latency_ms'].mean() if 'latency_ms' in df and df['latency_ms'].notna().any() else None,
        'unique_tools_used': df['tool_name'].nunique(),
    }

    # By tool
    by_tool = df.groupby('tool_name').agg(
        call_count=('tool_name', 'size'),
        success_count=('status', lambda x: (x == 'success').sum()),
        failure_count=('status', lambda x: (x != 'success').sum()),
        average_latency_ms=('latency_ms', 'mean')
    ).reset_index()
    by_tool = by_tool.sort_values(by='call_count', ascending=False)

    # By session
    by_session = df.groupby('session_id').agg(
        call_count=('session_id', 'size'),
        success_count=('status', lambda x: (x == 'success').sum()),
        failure_count=('status', lambda x: (x != 'success').sum()),
        average_latency_ms=('latency_ms', 'mean')
    ).reset_index()

    return summary, df, by_tool, by_session

def generate_report(summary, by_tool, outdir):
    """
    Generate a human-readable Markdown report.
    """
    report_path = os.path.join(outdir, 'report.md')
    with open(report_path, 'w') as f:
        f.write("# Tool Usage Analysis Report\n\n")
        f.write("## Overall Summary\n\n")
        f.write(f"- **Total Tool Calls:** {summary.get('total_tool_calls', 0)}\n")
        f.write(f"- **Successful Calls:** {summary.get('successful_calls', 0)}\n")
        f.write(f"- **Failed Calls:** {summary.get('failed_calls', 0)}\n")
        avg_latency = summary.get('average_latency_ms')
        if avg_latency is not None:
            f.write(f"- **Average Latency:** {avg_latency:.2f} ms\n")
        f.write(f"- **Unique Tools Used:** {summary.get('unique_tools_used', 0)}\n\n")
        
        f.write("## Top 10 Tools by Usage\n\n")
        f.write(by_tool.head(10).to_markdown(index=False))
        f.write("\n")

    logging.info(f"Generated Markdown report at {report_path}")

def pick_baseline_log(baseline_dir, baseline_name):
    """
    Pick a primary log file for a baseline directory.
    Preference order:
      1) files named like {baseline_name}_*.json or .jsonl (excluding tool_calls_*)
      2) any .json or .jsonl (excluding tool_calls_*)
    If multiple matches, choose the most recently modified.
    """
    preferred = []
    preferred.extend(glob(os.path.join(baseline_dir, f"{baseline_name}_*.json")))
    preferred.extend(glob(os.path.join(baseline_dir, f"{baseline_name}_*.jsonl")))
    preferred = [p for p in preferred if not os.path.basename(p).startswith("tool_calls_")]

    if not preferred:
        candidates = glob(os.path.join(baseline_dir, "*.json"))
        candidates += glob(os.path.join(baseline_dir, "*.jsonl"))
        candidates = [p for p in candidates if not os.path.basename(p).startswith("tool_calls_")]
    else:
        candidates = preferred

    if not candidates:
        return None

    return max(candidates, key=lambda p: os.path.getmtime(p))

def run_single(input_path, outdir):
    os.makedirs(outdir, exist_ok=True)

    events, log_format = parse_log_file(input_path)
    if not events:
        logging.warning("No events found in the log file.")
        return False

    tool_calls = extract_tool_calls(events)
    if not tool_calls:
        logging.warning("No tool calls found in the log file.")
        return False

    summary, df_calls, df_by_tool, df_by_session = analyze_tool_usage(tool_calls)

    # Save artifacts
    summary_path = os.path.join(outdir, 'tool_usage_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logging.info(f"Saved overall summary to {summary_path}")

    calls_csv_path = os.path.join(outdir, 'tool_calls.csv')
    df_calls.to_csv(calls_csv_path, index=False)
    logging.info(f"Saved detailed tool calls to {calls_csv_path}")

    by_tool_csv_path = os.path.join(outdir, 'tool_usage_by_tool.csv')
    df_by_tool.to_csv(by_tool_csv_path, index=False)
    logging.info(f"Saved tool usage by tool to {by_tool_csv_path}")

    by_session_csv_path = os.path.join(outdir, 'tool_usage_by_session.csv')
    df_by_session.to_csv(by_session_csv_path, index=False)
    logging.info(f"Saved tool usage by session to {by_session_csv_path}")

    generate_report(summary, df_by_tool, outdir)

    # Print console summary
    print("\n--- Tool Usage Summary ---")
    print(f"Log format detected: {log_format.upper()}")
    print(f"Total tool calls: {summary.get('total_tool_calls', 0)}")
    print("\nTop 10 tools by count:")
    print(df_by_tool[['tool_name', 'call_count']].head(10).to_string(index=False))
    print(f"\nOutputs written to: {os.path.abspath(outdir)}")
    print("--- End of Summary ---")
    return True

def run_batch(input_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    baseline_dirs = [
        d for d in glob(os.path.join(input_root, "*"))
        if os.path.isdir(d) and os.path.basename(d) != "analysis"
    ]

    if not baseline_dirs:
        logging.warning(f"No baseline directories found under {input_root}")
        return

    for baseline_dir in sorted(baseline_dirs):
        baseline_name = os.path.basename(baseline_dir)
        input_path = pick_baseline_log(baseline_dir, baseline_name)
        if not input_path:
            logging.warning(f"No suitable log file found for baseline: {baseline_name}")
            continue

        outdir = os.path.join(out_root, baseline_name)
        logging.info(f"Analyzing baseline: {baseline_name}")
        logging.info(f"Input: {input_path}")
        logging.info(f"Output: {outdir}")
        run_single(input_path, outdir)

def main():
    """
    Main function to run the analysis.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--input',
        default='./logs/run.jsonl',
        help='Path to the input log file (JSON or JSONL).'
    )
    parser.add_argument(
        '--outdir',
        default='./analysis/tool_usage',
        help='Directory to save the output artifacts for single-log analysis.'
    )
    parser.add_argument(
        '--input-root',
        default=None,
        help='Root directory containing baseline subdirectories to analyze in batch.'
    )
    parser.add_argument(
        '--out-root',
        default=None,
        help='Root directory to store batch analysis outputs (one folder per baseline).'
    )
    args = parser.parse_args()

    if args.input_root or args.out_root:
        input_root = args.input_root or './logs/chexagentbench'
        out_root = args.out_root or './logs/chexagentbench/analysis'
        run_batch(input_root, out_root)
    else:
        run_single(args.input, args.outdir)

if __name__ == '__main__':
    main()
