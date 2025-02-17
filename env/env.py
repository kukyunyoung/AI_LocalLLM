from dotenv import load_dotenv
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# API KEY 정보로드
load_dotenv()

def get_api_key():
    return os.getenv('OPENAI_API_KEY') 

def get_HFkey():
    return os.getenv('HUGGINGFACEHUB_API_TOKEN')

def os_path_exists(path):
    return os.path.exists(path)

def os_listdir(path):
    return os.listdir(path)

def os_path_join(path1, path2):
    return os.path.join(path1, path2)

def os_makedirs(path, exist_ok=False):
    return os.makedirs(path, exist_ok=exist_ok)

def os_path_dirname(path):
    return os.path.dirname(path)

def os_remove(path):
    return os.remove(path)