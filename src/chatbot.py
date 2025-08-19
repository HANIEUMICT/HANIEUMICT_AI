import json
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from config import OLLAMA_MODEL_NAME, SERVICE_HIERARCHY_PATH

# --- 프롬프트 템플릿 정의 ---
PROMPT_RECOMMEND = """
You are a precise and honest AI manufacturing consultant. Your task is to format the information from the [RETRIEVED CONTEXT] as a direct answer to the user's [QUESTION].

[INSTRUCTIONS]
1.  **Always start your answer with the user's original [QUESTION]** in the following format: `'{question}' 프로젝트에 대한 추천 서비스는 다음과 같습니다:`
2.  **Use the information from the [RETRIEVED CONTEXT]** to list the available services below the starting line.
3.  **CRITICAL RULE:** Format the list using Markdown line breaks (`\n`). Only include a line for a service if its value exists in the context and is not 'N/A' or empty. Use the exact Korean labels below.
    - `주서비스 : {{main_service}}`
    - `세부서비스: {{sub_service}}`
    - `재료: {{material}}`
4.  NEVER invent information.
5.  **ABSOLUTE FINAL RULE: Your response must contain ONLY the answer itself, without any additional explanations, comments, or notes like "참고" , "신뢰도", etc.

---
[EXAMPLE]
CONTEXT:
- project_description: 정밀 측정 장비용 광학 렌즈
- main_service: 연마/폴리싱
- sub_service: N/A
- material: 유리/세라믹
QUESTION: 광학 렌즈
CORRECT ANSWER:
'광학 렌즈'에 대한 추천 서비스는 다음과 같습니다:
주서비스 : 조립 \n
재료 : 유리/세라믹
---

[RETRIEVED CONTEXT]
{context}
---
[QUESTION]
{question}
---
[ANSWER (in Korean)]
"""

PROMPT_EXPLAIN = """
You are a helpful AI assistant. Your job is to clearly explain the service the user asked about, based on the [Context].

[INSTRUCTIONS]
1.  Read the [Context] to understand the service.
2.  If the context describes a sub-service, also mention which main service it belongs to.
3.  **You MUST summarize the entire explanation in 3 sentences or less.**
4.  Answer in a natural and friendly tone in Korean.

---
[Context]
{context}
---
[Question]
{question}
---
[ANSWER (in Korean)]
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
        """최대 3개의 프로젝트 문서 정보를 구조화된 텍스트로 변환합니다."""
        if not docs: return ""
        
        formatted_strings = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            formatted_strings.append(
                f"후보 {i+1}:\n"
                f"- project_description: {metadata.get('project_description', 'N/A')}\n"
                f"- main_service: {metadata.get('main_service', 'N/A')}\n"
                f"- sub_service: {metadata.get('sub_service', 'N/A')}\n"
                f"- material: {metadata.get('material', 'N/A')}"
            )
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
            inputs = {"context": context, "question": query}
            return self.explain_chain.invoke(inputs)
            
        return "오류: 잘못된 모드가 선택되었습니다."
            
        
        