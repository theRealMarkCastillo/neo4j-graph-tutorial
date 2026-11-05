"""What the extraction LLM returns after each exchange (validated with Pydantic)."""
from typing import Literal

from pydantic import BaseModel, Field

EntityKind = Literal["person", "pet", "place", "organization", "activity", "thing"]
MemoryKind = Literal["fact", "preference", "event", "plan", "feeling", "boundary"]


class EntityOut(BaseModel):
    name: str = Field(description="Canonical name, e.g. 'Maya' or 'Portland'")
    kind: EntityKind
    aliases: list[str] = Field(default_factory=list, description="Other names used, e.g. 'my sister'")
    summary: str = Field(default="", description="One short line describing it")


class RelationOut(BaseModel):
    source: str = Field(description="'user' for the user, otherwise an entity name")
    target: str = Field(description="An entity name")
    type: str = Field(description="snake_case relation, e.g. sister, partner, works_at, lives_in, owns")
    since: str | None = Field(default=None, description="ISO date or year if stated")


class MemoryOut(BaseModel):
    text: str = Field(description="Standalone third-person statement, e.g. 'Alex's sister Maya lives in Denver'")
    kind: MemoryKind
    about: list[str] = Field(default_factory=list, description="'user' and/or entity names it concerns")
    importance: int = Field(ge=1, le=10, description="1 trivia ... 10 core identity / major life event")
    occurred_on: str | None = Field(default=None, description="ISO date when an event happened")
    due_on: str | None = Field(default=None, description="ISO date a plan is due")
    supersedes: list[str] = Field(default_factory=list, description="ids of existing memories this replaces")


class Extraction(BaseModel):
    entities: list[EntityOut] = Field(default_factory=list)
    relations: list[RelationOut] = Field(default_factory=list)
    memories: list[MemoryOut] = Field(default_factory=list)
    invalidate: list[str] = Field(default_factory=list, description="ids of memories no longer true, with no replacement")
    completed_plans: list[str] = Field(default_factory=list, description="ids of plan memories that have now happened or been cancelled")


class EpisodeSummary(BaseModel):
    summary: str = Field(description="3-5 sentences: what was discussed and anything left open")
    mood: str = Field(description="The user's overall mood in a few words")
