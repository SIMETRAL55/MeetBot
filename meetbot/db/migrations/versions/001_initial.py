"""Initial schema baseline

Revision ID: 001_initial
Revises: None
Create Date: 2026-03-15

Baseline migration that represents the existing MeetBot schema.
This does not create tables (they already exist via create_all);
it just marks the initial state so future migrations can build on it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Tables already exist via Base.metadata.create_all()
    # This migration just establishes the baseline for Alembic tracking.
    #
    # Tables: users, jobs, segments, chat_sessions, chat_messages
    pass


def downgrade() -> None:
    # Cannot downgrade from baseline
    pass
