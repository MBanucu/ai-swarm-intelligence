import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_FILE = os.path.join(ROOT_DIR, "logs", "current_gen.json")
BASE_CODE = os.path.join(ROOT_DIR, "base_code")
ARCHIVE_DIR = os.path.join(ROOT_DIR, "logs", "archived_agents")
BASE_TEMPLATE = os.path.join(ROOT_DIR, ".opencode", "agents", "base_template.md")
BENCHMARK_HISTORY = os.path.join(ROOT_DIR, "logs", "benchmark_history.json")
IMPROVEMENT_DIR = os.path.join(ROOT_DIR, "improvement_suggestions")

MAX_RETRIES = 10
PRIMARY_MODEL = "opencode/deepseek-v4-flash-free"
FALLBACK_MODEL = "opencode-go/deepseek-v4-flash"
