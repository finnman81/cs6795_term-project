"""Logging utilities for experiment data."""

import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import sys


def init_log_file(path: Path) -> None:
    """
    Initialize log file with CSV headers if it doesn't exist.
    
    Args:
        path: Path to log file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        return  # File already exists
    
    headers = [
        "timestamp",
        "participant_id",
        "condition",
        "task_id",
        "query",
        "answer",
        "retrieved_chunks_count",
        "retrieved_chunks_titles",
        "start_time",
        "end_time",
        "duration_seconds",
        "extra_meta"
    ]
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def log_interaction(
    participant_id: str,
    condition: str,
    task_id: Optional[int],
    query: str,
    answer: str,
    retrieved_chunks: List,
    start_time: datetime,
    end_time: datetime,
    log_path: Path,
    extra_meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an interaction to the CSV file.
    
    Args:
        participant_id: Participant identifier
        condition: "baseline" or "cl_aware"
        task_id: Task ID from tasks.json (None for free-form queries)
        query: User query
        answer: LLM answer
        retrieved_chunks: List of RetrievedChunk objects
        start_time: Start timestamp
        end_time: End timestamp
        log_path: Path to log file
        extra_meta: Optional additional metadata (will be JSON stringified)
    """
    log_path = Path(log_path)
    init_log_file(log_path)
    
    # Calculate duration
    duration = (end_time - start_time).total_seconds()
    
    # Extract chunk information
    chunk_titles = [chunk.chunk.title for chunk in retrieved_chunks]
    
    # Format extra metadata
    extra_meta_str = ""
    if extra_meta:
        import json
        extra_meta_str = json.dumps(extra_meta)
    
    row = [
        datetime.now().isoformat(),
        participant_id,
        condition,
        task_id if task_id is not None else "",
        query,
        answer,
        len(retrieved_chunks),
        "; ".join(chunk_titles),
        start_time.isoformat(),
        end_time.isoformat(),
        duration,
        extra_meta_str
    ]
    
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def log_error(error_info: str) -> None:
    """
    Log error to stderr.
    
    Args:
        error_info: Error message or information
    """
    timestamp = datetime.now().isoformat()
    print(f"[ERROR {timestamp}] {error_info}", file=sys.stderr)

