from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_oauth_token: str = Field(default="", alias="ANTHROPIC_OAUTH_TOKEN")
    # Path to the Claude Code CLI. The SDK ships a Bun-compiled binary that
    # requires AVX CPU support; on hosts without AVX it crashes (SIGILL), so
    # point this at the npm Node.js CLI instead. Empty = auto-discover on PATH.
    claude_cli_path: str = Field(default="", alias="CLAUDE_CLI_PATH")
    mailto: str = Field(default="eterrrii@gmail.com", alias="MAILTO")
    cache_dir: Path = Field(default=Path("webapp_cache"), alias="CACHE_DIR")
    db_url: str = Field(default="sqlite+aiosqlite:///./webapp.db", alias="DB_URL")
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        alias="EMBEDDING_MODEL",
    )
    snapshot_id: str = Field(default="", alias="SNAPSHOT_ID")
    bertrend_db: Path = Field(
        default=Path("bertrend_emb_extracted/bertrend_emb/bertrend.db"),
        alias="BERTREND_DB",
    )

    openalex_base: str = "https://api.openalex.org"

    @property
    def taxonomy_cache(self) -> Path:
        return self.cache_dir / "taxonomy.json"


settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
