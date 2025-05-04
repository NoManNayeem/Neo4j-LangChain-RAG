# config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# point to your .env in the project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# now expose your settings
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
NEO4J_URI           = os.getenv("NEO4J_URI")
NEO4J_USERNAME      = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD      = os.getenv("NEO4J_PASSWORD")
AURA_INSTANCEID     = os.getenv("AURA_INSTANCEID")
AURA_INSTANCENAME   = os.getenv("AURA_INSTANCENAME")

# sanity‐check (optional)
required = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USERNAME": NEO4J_USERNAME,
    "NEO4J_PASSWORD": NEO4J_PASSWORD,
}
missing = [k for k,v in required.items() if not v]
if missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
