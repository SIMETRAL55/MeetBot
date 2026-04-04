"""Add OTP columns to users table.

Revision ID: 007_add_otp_columns
Revises: 006_add_email_auth
Create Date: 2026-04-02
"""

from alembic import op
import sqlalchemy as sa

revision = "007_add_otp_columns"
down_revision = "006_add_email_auth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("otp_hash", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("otp_expires", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("otp_purpose", sa.String(16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("otp_purpose")
        batch_op.drop_column("otp_expires")
        batch_op.drop_column("otp_hash")
