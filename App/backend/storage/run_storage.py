"""Storage manager for ABAC Pipeline Runs."""

import json
import sqlite3
import uuid
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class RunStorage:
    """Manages persistence of runs (metadata, inputs, outputs, errors)."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.runs_dir = self.storage_dir / "runs"
        self.db_path = self.storage_dir / "index.db"
        
        self._init_storage()
        
    def _init_storage(self):
        """Create directories and SQLite table if they don't exist."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model TEXT,
                embedding TEXT,
                status TEXT,
                total_records INTEGER DEFAULT 0,
                successful INTEGER DEFAULT 0,
                failed INTEGER DEFAULT 0,
                storage_path TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        
    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:4]
        return f"RUN_{timestamp}_{short_uuid}"
        
    def create_run(self, run_type: str, model_info: Dict[str, str], initial_metadata: Dict[str, Any] = None) -> str:
        """Initialize a new Run in database and filesystem."""
        run_id = self._generate_run_id()
        created_at = datetime.now(timezone.utc).isoformat()
        
        run_path = self.runs_dir / run_id
        run_path.mkdir(parents=True, exist_ok=False)
        
        # Save initial metadata
        metadata = {
            "run_id": run_id,
            "run_type": run_type,
            "created_at": created_at,
            "model": model_info,
            "status": "created",
            **(initial_metadata or {})
        }
        
        self._write_json(run_path / "metadata.json", metadata)
        
        # Insert to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO runs (run_id, run_type, created_at, model, embedding, status, storage_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, run_type, created_at, 
            model_info.get("architecture", ""), 
            model_info.get("embedding", ""), 
            "created", str(run_path)
        ))
        conn.commit()
        conn.close()
        
        return run_id
        
    def update_status(self, run_id: str, status: str, metrics: Dict[str, int] = None):
        """Update run status and metrics in both SQLite and metadata.json."""
        run_path = self.runs_dir / run_id
        meta_file = run_path / "metadata.json"
        
        metadata = self.get_metadata(run_id) or {}
        metadata["status"] = status
        
        if metrics:
            metadata.setdefault("processing", {}).update(metrics)
            
        self._write_json(meta_file, metadata)
        
        # Update DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if metrics:
            cursor.execute('''
                UPDATE runs SET status = ?, total_records = ?, successful = ?, failed = ?
                WHERE run_id = ?
            ''', (
                status, 
                metrics.get("total_records", 0), 
                metrics.get("successful", 0), 
                metrics.get("failed", 0), 
                run_id
            ))
        else:
            cursor.execute('UPDATE runs SET status = ? WHERE run_id = ?', (status, run_id))
        conn.commit()
        conn.close()

    def append_jsonl(self, run_id: str, filename: str, data: Dict[str, Any]):
        """Append a single record to a JSONL file."""
        file_path = self.runs_dir / run_id / filename
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    def save_json(self, run_id: str, filename: str, data: Any):
        """Save entire JSON file atomically."""
        file_path = self.runs_dir / run_id / filename
        self._write_json(file_path, data)

    def _write_json(self, path: Path, data: Any):
        """Atomic write to prevent corruption."""
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
        
    def get_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        meta_file = self.runs_dir / run_id / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
        
    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List runs from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM runs ORDER BY created_at DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
        
    def delete_run(self, run_id: str):
        """Delete run from database and filesystem."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM runs WHERE run_id = ?', (run_id,))
        conn.commit()
        conn.close()
        
        run_path = self.runs_dir / run_id
        if run_path.exists():
            shutil.rmtree(run_path, ignore_errors=True)
            
    def get_file_path(self, run_id: str, filename: str) -> Optional[Path]:
        """Get absolute path to a specific run artifact."""
        path = self.runs_dir / run_id / filename
        return path if path.exists() else None
