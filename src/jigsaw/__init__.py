import logging
import dotenv
import os
import sys
from datetime import datetime
import warnings

dotenv.load_dotenv()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

PROJECT_NAME = __name__.split(".")[-1]
__version__ = "0.25.0"

logging_str: str = "%(asctime)s [%(levelname)s] : %(module)s : %(message)s"
LOG_DIR = "logs"
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_filepath = os.path.join(LOG_DIR, f"{timestamp}.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("JigsawLogger")
