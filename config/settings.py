from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    groq_api_key: str = Field(
        ...,
        description="Groq API Key for LLM"
    )

    embedding_model: str = Field(
        default="jina-embeddings-v5-text-small",
        description="model used to generate the embeddings"
    )

    chat_model: str = Field(
        default="openai/gpt-oss-120b",
        description="model used for chat reasoning"
    )

    langsmith_api_key: str = Field(
        ...,
        description="langsmith api key for agent tracing"
    )

    langsmith_project: str = Field(
        default="clinical_trial_intelligence",
        description="Langsmith Project Name"
    )

    langsmith_tracing_v2: str = Field(
        default=True,
        description="enable langsmith tracing for all agent runs"
    )

    gcp_project_id: str = Field(
        ...,
        description="gcp project id"
    )

    gcp_region: str = Field(
        default="us-central1",
        description="gcp region for all cloud resources"
    )

    gcp_bucket_name: str = Field(
        ...,
        description="google cloud storage bucket name"
    )

    db_host: str = Field(
        ...,
        description="cloud sql host ip (local) or socket path (cloud run)"
    )

    db_port: str = Field(
        default=5432,
        description="postgresql port"
    )

    db_name: str = Field(
        default="clinical_trial_db",
        description="PostgreSQL database name"
    )

    db_user: str = Field(
        ...,
        description="PostgreSQL database user"
    )

    db_password: str = Field(
        ...,
        description="PostgreSQL database password"
    )

    clinical_trials_base_url: str = Field(
        default="https://clinicaltrials.gov/api/v2",
        description="clinicaltrials.gov api v2 base url"
    )

    clinical_trials_page_size: int = Field(
        default=100,
        description="number of studies to fetch per api page"
    )

    pubmed_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        description="Pubmed eutils api base url"
    )

    api_host: str = Field(
        default="0.0.0.0",
        description="fast api host address"
    )

    api_port: int = Field(
        default=8000,
        description="fast api port"
    )

    api_env: str = Field(
        default="development",
        description="Environment name: development or production"
    )

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://"
            f"{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}"
            f"/{self.db_name}"
        )
    
    @property
    def is_production(self)->bool:
        return self.api_env.lower() == "production"

settings = Settings()