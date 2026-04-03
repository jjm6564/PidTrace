import os

from dotenv import load_dotenv


load_dotenv()


MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")  # ollama / openai / anthropic
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")

INPUT_IMAGE_PATH = os.getenv("INPUT_IMAGE_PATH", "data/input/sample_pid.png")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output")
TEMP_DIR = os.getenv("TEMP_DIR", "data/temp")
