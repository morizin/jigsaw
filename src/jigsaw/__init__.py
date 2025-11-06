import logging
import dotenv
import os, sys
from datetime import datetime
import warnings

dotenv.load_dotenv()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

PROJECT_NAME = __name__.split(".")[-1]

logging_str: str = "%(asctime)s [%(levelname)s] : %(module)s : %(message)s"
log_dir = "logs"
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_filepath = os.path.join(log_dir, f"{timestamp}.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("JigsawLogger")
