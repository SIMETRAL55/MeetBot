"""Add pageindex columns to jobs table.

Revision ID: 002_add_pageindex_columns
Revises: 001_initial
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "002_add_pageindex_columns"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.add_column(sa.Column("pageindex_path", sa.String(500), nullable=True))
        batch_op.add_column(sa.Column("pageindex_status", sa.String(20), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_column("pageindex_status")
        batch_op.drop_column("pageindex_path")
