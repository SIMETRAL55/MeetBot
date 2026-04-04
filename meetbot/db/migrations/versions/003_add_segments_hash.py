"""Add segments_hash column to jobs table.

Revision ID: 003_add_segments_hash
Revises: 002_add_pageindex_columns
Create Date: 2026-03-26
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "003_add_segments_hash"
down_revision = "002_add_pageindex_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.add_column(
            sa.Column("segments_hash", sa.String(64), nullable=True, server_default="")
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_column("segments_hash")
