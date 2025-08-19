import streamlit as st
import requests

st.title("제조 전문가 AI 챗봇")

if "mode" not in st.session_state:
    st.session_state.mode = None
    st.session_state.messages = []

# --- 모드 선택 화면 ---
if st.session_state.mode is None:
    st.write("안녕하세요! 어떤 도움이 필요하신가요?")
    if st.button("1. 서비스 추천"):
        st.session_state.mode = "recommend"
        # 챗봇의 첫 메시지를 대화 기록에 추가
        st.session_state.messages.append({"role": "assistant", "content": "어떤 제품에 대한 서비스 추천을 원하시나요?"})
        st.rerun() # 화면을 즉시 새로고침하여 채팅창 표시
    
    if st.button("2. 서비스 설명"):
        st.session_state.mode = "explain"
        st.session_state.messages.append({"role": "assistant", "content": "알고 싶은 서비스를 말씀해주세요."})
        st.rerun()

# --- 채팅 화면 ---
else:
    if st.button("◀️ 모드 선택으로 돌아가기"):
        st.session_state.mode = None
        st.session_state.messages = []
        st.rerun()
        
    # 이전 대화 내용 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                api_url = "http://127.0.0.1:8000/chat"
                response = requests.post(
                    api_url, 
                    json={"query": prompt, "mode": st.session_state.mode}
                )
                response.raise_for_status()
                answer = response.json().get("answer", "오류가 발생했습니다.")
                message_placeholder.markdown(answer)
            except requests.exceptions.RequestException as e:
                answer = f"백엔드 서버에 연결할 수 없습니다: {e}"
                message_placeholder.markdown(answer)
            
        st.session_state.messages.append({"role": "assistant", "content": answer})