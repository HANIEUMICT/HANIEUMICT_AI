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
        
    def add_project(self, project_data: dict) -> bool:
        #성공 시 True, 중복 시 False를 반환

        content = (f"프로젝트 '{project_data['project_description']}'의 주 서비스는 {project_data['main_service']}, "
                   f"세부 서비스는 {project_data['sub_service']}, 재료는 {project_data['material']}입니다.")
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        existing_docs = self.project_db.get(ids=[content_hash])
        if existing_docs and existing_docs['ids']:
            return False

        metadata = {
            'id': content_hash, 
            'project_description': str(project_data['project_description']),
            'main_service': str(project_data['main_service']),
            'sub_service': str(project_data['sub_service']),
            'material': str(project_data['material'])
        }
        document = Document(page_content=content, metadata=metadata)
        
        self.project_db.add_documents([document], ids=[content_hash])
        return True
    
    def update_project_db(self):
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
            df = pd.read_csv(SERVICE_CSV_PATH, encoding='utf-8').fillna('') 
        except FileNotFoundError:
            print(f"오류: {SERVICE_CSV_PATH} 파일을 찾을 수 없습니다.")
            return
        
        documents = []
        for _, row in df.iterrows():
            if row['sub_service']: 
                content = f"주 서비스 '{row['main_service']}'의 세부 서비스인 '{row['sub_service']}'에 대한 설명: {row['description']}"
                metadata = {'service_name': row['sub_service'], 'parent_service': row['main_service']}
            else: 
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
            return self.service_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'k': 1, 'score_threshold': SCORE_THRESHOLD}
            )
