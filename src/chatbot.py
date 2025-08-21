import json
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from config import OLLAMA_MODEL_NAME, SERVICE_HIERARCHY_PATH

PROMPT_RECOMMEND = """
You are a precise and honest AI manufacturing consultant. Your task is to analyze up to 3 [RETRIEVED CONTEXT] and provide the most helpful recommendation based on the user's [QUESTION].
[INSTRUCTIONS]
1.  Start your answer with: `'{question}'에 대한 추천 서비스는 다음과 같습니다.`
    - Then, list the available information from the context.
     -Only include a line for a service (`주서비스`, `세부서비스`, `재료`) if its value exists in the context.
2.  NEVER invent information.
3.  Your response must contain ONLY the answer itself, without any additional explanations, comments, or notes like "참고" , "신뢰도", etc.

---
[EXAMPLE]
CONTEXT:
- project_description: 정밀 측정 장비용 광학 렌즈
- main_service: 연마/폴리싱
- material: 유리/세라믹
QUESTION: 광학 렌즈
CORRECT ANSWER:
'광학 렌즈'에 대한 추천 서비스는 다음과 같습니다.
주서비스 : 연마/폴리싱
재료: 유리/세라믹
---
[RETRIEVED CONTEXT]
{context}
---
[QUESTION]
{question}
---
[ANSWER (in Korean only, like the examples, with line breaks, without any other notes or labels)]
"""

PROMPT_EXPLAIN = """
You are an AI assistant that explains concepts. Your task is to explain the following [CONTEXT] in exactly one clear and concise Korean sentences.
Your response must contain ONLY the answer itself, without comments, or notes like "참고" , "신뢰도", etc.
---
[CONTEXT]
{context}
---
[ANSWER (in Korean only, without any other notes or labels like "참고" "신뢰도")]
"""



class Chatbot:
    def __init__(self, project_retriever, service_retriever):
        self.project_retriever = project_retriever
        self.service_retriever = service_retriever
        self.llm = ChatOllama(model=OLLAMA_MODEL_NAME)

        self.recommend_chain: Runnable = (
            PromptTemplate.from_template(PROMPT_RECOMMEND)
            | self.llm
            | StrOutputParser()
        )
        self.explain_chain: Runnable = (
            PromptTemplate.from_template(PROMPT_EXPLAIN)
            | self.llm
            | StrOutputParser()
        )
        #self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=False)
        #self.last_full_context = None
        

    def _get_rag_chain(self, prompt_template: str) -> Runnable:
        prompt = PromptTemplate.from_template(prompt_template)
        return prompt | self.llm | StrOutputParser()

    def _format_project_docs(self, docs: list) -> str:
        if not docs:
            return ""

        formatted_strings = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            
            parts = [f"--- 후보 {i+1} ---"]
            parts.append(f"- project_description: {metadata.get('project_description', 'N/A')}")

            main_service = metadata.get('main_service')
            if main_service and main_service not in ['N/A', '해당 없음', '']:
                parts.append(f"  main_service: {main_service}")

            sub_service = metadata.get('sub_service')
            if sub_service and sub_service not in ['N/A', '해당 없음', '']:
                parts.append(f"  sub_service: {sub_service}")
                
            material = metadata.get('material')
            if material and material not in ['N/A', '해당 없음', '']:
                parts.append(f"  material: {material}")

            formatted_strings.append("\n".join(parts))

        return "\n\n".join(formatted_strings)

    def generate_response(self, query: str, mode: str) -> str:
        if mode == "recommend":
            retrieved_docs = self.project_retriever.invoke(query)
            if not retrieved_docs:
                return "죄송합니다. 관련 정보를 찾을 수 없습니다."
            
            context = self._format_project_docs(retrieved_docs)
            inputs = {"context": context, "question": query}
            return self.recommend_chain.invoke(inputs)


        elif mode == "explain":
            retrieved_docs = self.service_retriever.invoke(query)
            if not retrieved_docs:
                return "죄송합니다. 관련 정보를 찾을 수 없습니다."
            
            context = retrieved_docs[0].page_content
            inputs = {"context": context}
            return self.explain_chain.invoke(inputs)
            
        return "오류: 잘못된 모드가 선택되었습니다."
            
        
        