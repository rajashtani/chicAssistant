#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

DB_ROOT = os.environ.get('DB_ROOT')

# Chroma settings
CLIENT_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=DB_ROOT,
        anonymized_telemetry=False
)