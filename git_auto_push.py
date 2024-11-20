import os
import subprocess
import datetime

commit_message = f"Auto-commit: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

try:
    subprocess.run(["git", "add", "."], check=True)

    subprocess.run(["git", "commit", "-m", commit_message], check=True)

    subprocess.run(["git", "push", "origin", "main"], check=True)

    print("Changes committed and pushed successfully!")

except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
