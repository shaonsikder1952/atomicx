"""Run Alembic migrations to upgrade database schema."""

import subprocess
import sys

def main():
    """Run alembic upgrade head."""
    try:
        # Try running via python -m alembic
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd="."
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"\n❌ Migration failed with exit code {result.returncode}")
            sys.exit(1)
        else:
            print("\n✅ Migration completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running migration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
