"""What the extraction LLM returns after each exchange (validated with Pydantic)."""
from typing import Literal

from pydantic import BaseModel, Field

EntityKind = Literal["person", "pet", "place", "organization", "activity", "thing"]
MemoryKind = Literal["fact", "preference", "event", "plan", "feeling", "boundary"]


class EntityOut(BaseModel):
    name: str = Field(description="Canonical name, e.g. 'Maya' or 'Portland'")
    kind: EntityKind
    aliases: list[str] = Field(default_factory=list, description="Other names used, e.g. 'my sister'")
    summary: str = Field(default="", description="One short line describing it, e.g. 'Alex's cousin, lives in Ohio'")
    ref: str | None = Field(default=None, description="id of the KNOWN entity this is (e.g. 'e_1a2b3c'), or null if it is not known yet")
    new: bool = Field(default=False, description="true if this is a DIFFERENT entity that shares a name with a known one (e.g. a second Sam)")


class RelationOut(BaseModel):
    source: str = Field(description="'user' for the user, otherwise an entity name or id")
    target: str = Field(description="An entity name or id")
    type: str = Field(description="snake_case relation, e.g. sister, partner, ex, works_at, lives_in, owns")
    since: str | None = Field(default=None, description="ISO date or year if stated")
    ended: bool = Field(default=False, description="true if this relation has ENDED (breakup, moved away, left a job)")


class MemoryOut(BaseModel):
    text: str = Field(description="Standalone third-person statement, e.g. 'Alex's sister Maya lives in Denver'")
    kind: MemoryKind
    about: list[str] = Field(default_factory=list, description="'user' and/or entity names or ids it concerns")
    importance: int = Field(ge=1, le=10, description="1 trivia ... 10 core identity / major life event")
    occurred_on: str | None = Field(default=None, description="ISO date when an event happened")
    due_on: str | None = Field(default=None, description="ISO date a plan is due")
    annual_date: str | None = Field(default=None, description="'MM-DD' for dates that repeat every year (birthdays, anniversaries)")
    sensitive: bool = Field(default=False, description="true for health, mental health, sexuality, religion, finances, legal trouble or trauma")
    supersedes: list[str] = Field(default_factory=list, description="ids of existing memories this replaces")


class Extraction(BaseModel):
    entities: list[EntityOut] = Field(default_factory=list)
    relations: list[RelationOut] = Field(default_factory=list)
    memories: list[MemoryOut] = Field(default_factory=list)
    invalidate: list[str] = Field(default_factory=list, description="ids of memories no longer true, with no replacement")
    completed_plans: list[str] = Field(default_factory=list, description="ids of plan memories that have now happened or been cancelled")
    forget_entities: list[str] = Field(default_factory=list, description="names or ids of people/things the user EXPLICITLY asked to be forgotten entirely")


class EpisodeSummary(BaseModel):
    summary: str = Field(description="3-5 sentences: what was discussed and anything left open")
    mood: str = Field(description="The user's overall mood in a few words")
