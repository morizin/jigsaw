import logging
import dotenv
import os, sys
import warnings
dotenv.load_dotenv()
warnings.filterwarnings('ignore')

logging_str : str = "%(asctime)s [%(levelname)s] : %(module)s : %(message)s"
log_dir = 'logs'
log_filepath = os.path.join(log_dir, "logging.log")
os.makedirs(log_dir, exist_ok = True)

logging.basicConfig(
        level = logging.INFO, 
        format = logging_str,
        handlers = [
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
)

logger = logging.getLogger("JigsawLogger")
