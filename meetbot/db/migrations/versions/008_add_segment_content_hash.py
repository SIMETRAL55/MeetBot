"""Add content_hash column to segments table.

Revision ID: 008_add_segment_content_hash
Revises: 007_add_otp_columns
Create Date: 2026-04-07
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "008_add_segment_content_hash"
down_revision = "007_add_otp_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("segments") as batch_op:
        batch_op.add_column(
            sa.Column("content_hash", sa.String(64), nullable=True, server_default="")
        )


def downgrade() -> None:
    with op.batch_alter_table("segments") as batch_op:
        batch_op.drop_column("content_hash")
