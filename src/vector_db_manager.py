import pandas as pd
import os
import hashlib
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from config import PROJECT_CSV_PATH, SERVICE_CSV_PATH, PROJECT_DB_DIRECTORY, SERVICE_DB_DIRECTORY, EMBEDDING_MODEL_NAME, RETRIEVER_K, SCORE_THRESHOLD

class VectorDBManager:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            #model_kwargs={"device": "cuda"}
        )

        self.project_db = Chroma(
            persist_directory=PROJECT_DB_DIRECTORY,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        self.service_db = Chroma(
            persist_directory=SERVICE_DB_DIRECTORY,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def update_project_db(self):
        """manufacturing_dataset.csv 파일로 프로젝트 DB를 업데이트합니다."""
        print("프로젝트 DB 업데이트를 시작합니다...")
        existing_ids = set(item['id'] for item in self.project_db.get(include=["metadatas"])['metadatas'] if 'id' in item)
        print(f"현재 프로젝트 DB에 저장된 문서는 총 {len(existing_ids)}개입니다.")

        try:
            df = pd.read_csv(PROJECT_CSV_PATH, encoding='utf-8')
        except FileNotFoundError:
            print(f"오류: {PROJECT_CSV_PATH} 파일을 찾을 수 없습니다.")
            return
        except UnicodeDecodeError:
            df = pd.read_csv(PROJECT_CSV_PATH, encoding='cp949')

        new_documents, new_ids = [], []
        for _, row in df.iterrows():
            content = (f"프로젝트 '{row['project_description']}'의 주 서비스는 {row['main_service']}, "
                       f"세부 서비스는 {row['sub_service']}, 재료는 {row['material']}입니다.")
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            if content_hash not in existing_ids:
                metadata = {
                    'id': content_hash, 'project_description': str(row['project_description']),
                    'main_service': str(row['main_service']), 'sub_service': str(row['sub_service']),
                    'material': str(row['material'])
                }
                new_documents.append(Document(page_content=content, metadata=metadata))
                new_ids.append(content_hash)
        
        if new_documents:
            print(f"새로운 프로젝트 문서 {len(new_documents)}개를 DB에 추가합니다.")
            self.project_db.add_documents(new_documents, ids=new_ids)
            print("프로젝트 DB 업데이트가 완료되었습니다.")
        else:
            print("새롭게 추가할 프로젝트 데이터가 없습니다.")

    def update_service_db(self, rebuild=False):
        if rebuild and os.path.exists(SERVICE_DB_DIRECTORY):
            print(f"기존 서비스 DB '{SERVICE_DB_DIRECTORY}'를 삭제하고 재생성합니다.")
            shutil.rmtree(SERVICE_DB_DIRECTORY)
            self.service_db = Chroma(
                persist_directory=SERVICE_DB_DIRECTORY,
                embedding_function=self.embedding_model,
                collection_metadata={"hnsw:space": "cosine"}
            )

        try:
            df = pd.read_csv(SERVICE_CSV_PATH, encoding='utf-8').fillna('') # 빈칸(NaN)을 빈 문자열로 처리
        except FileNotFoundError:
            print(f"오류: {SERVICE_CSV_PATH} 파일을 찾을 수 없습니다.")
            return
        
        documents = []
        for _, row in df.iterrows():
            # [핵심] 세부 서비스의 경우, 주 서비스의 맥락을 포함하여 content를 생성
            if row['sub_service']: # 세부 서비스인 경우
                content = f"주 서비스 '{row['main_service']}'의 세부 서비스인 '{row['sub_service']}'에 대한 설명: {row['description']}"
                metadata = {'service_name': row['sub_service'], 'parent_service': row['main_service']}
            else: # 주 서비스인 경우
                content = f"주 서비스 '{row['main_service']}'에 대한 설명: {row['description']}"
                metadata = {'service_name': row['main_service']}
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        if documents:
            self.service_db.add_documents(documents)
            print(f"총 {len(documents)}개의 서비스 정의를 DB에 저장 및 업데이트했습니다.")
        else:
            print("서비스 정의 데이터가 없습니다.")

    def get_retriever(self, mode : str):
        if mode == "recommend":
                return self.project_db.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={'k': RETRIEVER_K, 'score_threshold': SCORE_THRESHOLD}
                )
        elif mode == "explain":
            # 서비스 설명은 보통 정확한 명칭으로 검색하므로, k=1로 설정하여 가장 정확한 것 하나만 찾음
            return self.service_db.as_retriever(search_kwargs={'k': 1})
