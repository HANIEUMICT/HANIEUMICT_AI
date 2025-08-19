"""
설정 정보 담아둔 파일로

데이터셋 파일 경로
db 디렉토리 경로
서비스 계층 구조 파일 경로

임베딩 모델 이름
OLLAMA 모델 이름

검색 갯수 k 값
점수 임계값
"""


PROJECT_CSV_PATH = "./data/manufacturing_dataset.csv"
SERVICE_CSV_PATH = "./data/service_definitions.csv"
PROJECT_DB_DIRECTORY = "./project_db"
SERVICE_DB_DIRECTORY = "./service_db"
SERVICE_HIERARCHY_PATH = "./service.json"

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
#OLLAMA_MODEL_NAME = "gemma"
OLLAMA_MODEL_NAME = "chat_bot"

RETRIEVER_K = 3  
SCORE_THRESHOLD = 0.5