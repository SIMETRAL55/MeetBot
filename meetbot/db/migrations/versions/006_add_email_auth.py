"""Add email auth columns to users table.

Revision ID: 006_add_email_auth
Revises: 005_add_app_settings
Create Date: 2026-03-30
"""

from alembic import op
import sqlalchemy as sa

revision = "006_add_email_auth"
down_revision = "005_add_app_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("email", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("email_verified", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("firebase_uid", sa.String(128), nullable=True))
        batch_op.add_column(sa.Column("password_reset_token", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("password_reset_expires", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("failed_login_attempts", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("locked_until", sa.DateTime(), nullable=True))
        batch_op.create_index("ix_users_email", ["email"], unique=True)
        batch_op.create_index("ix_users_firebase_uid", ["firebase_uid"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_index("ix_users_firebase_uid")
        batch_op.drop_index("ix_users_email")
        batch_op.drop_column("locked_until")
        batch_op.drop_column("failed_login_attempts")
        batch_op.drop_column("password_reset_expires")
        batch_op.drop_column("password_reset_token")
        batch_op.drop_column("firebase_uid")
        batch_op.drop_column("email_verified")
        batch_op.drop_column("email")
