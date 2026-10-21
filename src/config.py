import os
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class PostgresVectorConfig:
    host: str = os.getenv("RAG_PG_HOST", "localhost")
    port: int = int(os.getenv("RAG_PG_PORT", "5434"))
    database: str = os.getenv("RAG_PG_DB", "rag_db")
    user: str = os.getenv("RAG_PG_USER", "rag_user")
    password: str = os.getenv("RAG_PG_PASSWORD", "rag_password")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

DB_CONFIG = PostgresVectorConfig()
