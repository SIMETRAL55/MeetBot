"""Add app_settings table for runtime-configurable settings.

Revision ID: 005_add_app_settings
Revises: 004_add_retrieval_method_to_message
Create Date: 2026-03-27

"""
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision = "005_add_app_settings"
down_revision = "004_add_retrieval_method_to_message"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(120), primary_key=True),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False, default=datetime.utcnow),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
