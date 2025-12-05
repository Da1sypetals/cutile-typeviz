#!/usr/bin/env python3
"""
Reset user.name and user.email for all commits in a Git repository.
Requires: git-filter-repo (install via: pip install git-filter-repo)
"""

import subprocess
import sys
import os


def check_git_filter_repo():
    """Check if git-filter-repo is installed."""
    try:
        subprocess.run(["git", "filter-repo", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def reset_all_authors(new_name, new_email, repo_path="."):
    """
    Reset author name and email for all commits.

    Args:
        new_name: The new author name
        new_email: The new author email
        repo_path: Path to the git repository (default: current directory)
    """

    # Check if git-filter-repo is installed
    if not check_git_filter_repo():
        print("ERROR: git-filter-repo is not installed.")
        print("Install it with: pip install git-filter-repo")
        sys.exit(1)

    # Check if we're in a git repository
    if not os.path.exists(os.path.join(repo_path, ".git")):
        print(f"ERROR: {repo_path} is not a git repository")
        sys.exit(1)

    # Warning message
    print("⚠️  WARNING: This will rewrite Git history!")
    print("   - All commit hashes will change")
    print("   - Force push will be required to update remote")
    print("   - Make a backup before proceeding")
    print()
    print(f"New author name:  {new_name}")
    print(f"New author email: {new_email}")
    print()

    try:
        # Run git filter-repo
        cmd = [
            "git",
            "filter-repo",
            "--force",
            "--commit-callback",
            f"""
commit.author_name = b'{new_name}'
commit.author_email = b'{new_email}'
commit.committer_name = b'{new_name}'
commit.committer_email = b'{new_email}'
""",
        ]

        subprocess.run(cmd, cwd=repo_path, check=True)

        print("\n✅ Successfully reset all authors!")
        print("\nNext steps:")
        print("1. Verify the changes with: git log")
        print("2. Force push to remote: git push --force --all")
        print("3. Force push tags: git push --force --tags")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configuration
    NEW_NAME = "Da1sypetals"
    NEW_EMAIL = "da1sypetals.iota@gmail.com"
    REPO_PATH = "."  # Current directory, or specify path

    # You can also get these from command line arguments
    if len(sys.argv) == 3:
        NEW_NAME = sys.argv[1]
        NEW_EMAIL = sys.argv[2]
    elif len(sys.argv) == 4:
        NEW_NAME = sys.argv[1]
        NEW_EMAIL = sys.argv[2]
        REPO_PATH = sys.argv[3]

    reset_all_authors(NEW_NAME, NEW_EMAIL, REPO_PATH)
