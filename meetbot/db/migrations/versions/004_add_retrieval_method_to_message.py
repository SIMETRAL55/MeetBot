"""Add retrieval_method column to chat_messages table.

Revision ID: 004_add_retrieval_method_to_message
Revises: 003_add_segments_hash
Create Date: 2026-03-26
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "004_add_retrieval_method_to_message"
down_revision = "003_add_segments_hash"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.add_column(
            sa.Column("retrieval_method", sa.String(20), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.drop_column("retrieval_method")
