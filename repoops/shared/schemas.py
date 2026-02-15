from pydantic import BaseModel, Field
from typing import Any, Optional, Dict

class RunCreate(BaseModel):
    repo_url: str = Field(..., description="Git URL to clone")
    ref: str = Field(default="main", description="Git ref/branch/sha")
    task_text: str = Field(..., description="Ticket/issue text")
    commands_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional commands profile: {install: 'pnpm i', test: 'pnpm test', lint: 'pnpm lint', build: 'pnpm build'}"
    )

class RunOut(BaseModel):
    id: str
    repo_url: str
    ref: str
    task_text: str
    status: str
    created_at: str
    updated_at: str
    error: Optional[str] = None

class ArtifactOut(BaseModel):
    kind: str
    content: str
    created_at: str
