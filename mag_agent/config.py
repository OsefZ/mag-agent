# mag_agent/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Load Graph DB Configuration (defaults for local Neo4j/Memgraph)
GRAPH_DB_URI = os.getenv("GRAPH_DB_URI", "bolt://localhost:7687")
GRAPH_DB_USER = os.getenv("GRAPH_DB_USER", "")
GRAPH_DB_PASSWORD = os.getenv("GRAPH_DB_PASSWORD", "")

# Load Logging Configuration
LOG_LEVEL = os.getenv("MAG_LOG_LEVEL", "INFO").upper()