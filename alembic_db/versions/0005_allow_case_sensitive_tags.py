"""
Allow case-sensitive tag names.

Revision ID: 0005_allow_case_sensitive_tags
Revises: 0004_drop_tag_type
Create Date: 2026-06-16
"""

from alembic import op

revision = "0005_allow_case_sensitive_tags"
down_revision = "0004_drop_tag_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite cannot ALTER/DROP CHECK constraints. Recreate the small tag
        # vocabulary table without the lowercase constraint while preserving
        # existing tag names.
        op.execute("PRAGMA foreign_keys=OFF")
        op.execute(
            "CREATE TABLE tags_new ("
            "name VARCHAR(512) NOT NULL, "
            "CONSTRAINT pk_tags PRIMARY KEY (name)"
            ")"
        )
        op.execute("INSERT INTO tags_new(name) SELECT name FROM tags")
        op.execute("DROP TABLE tags")
        op.execute("ALTER TABLE tags_new RENAME TO tags")
        op.execute("PRAGMA foreign_keys=ON")
        return

    op.drop_constraint("ck_tags_ck_tags_lowercase", "tags", type_="check")


def downgrade() -> None:
    # Existing mixed-case tags cannot satisfy the old constraint. Lowercase them
    # before restoring it, merging duplicate vocabulary/link rows that collide.
    op.execute("INSERT OR IGNORE INTO tags(name) SELECT lower(name) FROM tags")
    op.execute(
        "DELETE FROM asset_reference_tags "
        "WHERE rowid NOT IN ("
        "  SELECT MIN(rowid) FROM asset_reference_tags "
        "  GROUP BY asset_reference_id, lower(tag_name)"
        ")"
    )
    op.execute("UPDATE asset_reference_tags SET tag_name = lower(tag_name)")
    op.execute("DELETE FROM tags WHERE name != lower(name)")

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        op.execute("PRAGMA foreign_keys=OFF")
        op.execute(
            "CREATE TABLE tags_new ("
            "name VARCHAR(512) NOT NULL, "
            "CONSTRAINT pk_tags PRIMARY KEY (name), "
            "CONSTRAINT ck_tags_lowercase CHECK (name = lower(name))"
            ")"
        )
        op.execute("INSERT INTO tags_new(name) SELECT name FROM tags")
        op.execute("DROP TABLE tags")
        op.execute("ALTER TABLE tags_new RENAME TO tags")
        op.execute("PRAGMA foreign_keys=ON")
        return

    op.create_check_constraint(
        "ck_tags_ck_tags_lowercase", "tags", "name = lower(name)"
    )
