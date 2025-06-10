import base64
import streamlit as st  # 모든 streamlit 명령은 "st" alias로 사용할 수 있습니다.
import bedrock as glib  # 로컬 라이브러리 스크립트에 대한 참조
import time
# from langchain.callbacks import StreamlitCallbackHandler

##################### LocalTest ########################
class DummyCallback:
    def _complete_current_thought(self): pass

##################### Functions ########################
def parse_image(metadata, tag):
    if tag in metadata:
        st.image(base64.b64decode(metadata[tag]))


def parse_table(metadata, tag):
    if tag in metadata:
        st.markdown(metadata[tag], unsafe_allow_html=True)


def parse_metadata(metadata):
    # Image, Table 이 있을 경우 파싱해 출력
    category = "None"
    if "category" in metadata:
        category = metadata["category"]
        if category == "Table":
            # parse_table(metadata, "text_as_html") # 테이블 html은 이미지로 대체
            parse_image(metadata, "image_base64")
        elif category == "Image":
            parse_image(metadata, "image_base64")
        else:
            pass
    st.markdown(' - - - ')


def show_document_info_label():
    with st.container(border=True):
        if menu == "🤖 Chatbot":
            st.markdown('''**🔍 원하는 도면이 있나요?**''')
            st.markdown(
                '''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;👉 스파이럴 슈트 10개 있는 레이아웃 알려줘. 라고 질문해보세요.''')
            st.markdown("""""")
            st.markdown('''**📖 원하는 매뉴얼이 있나요?**''')
            st.markdown(
                '''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;👉 터치스크린의 조작법에 대해 알려줘. 라고 질문해보세요.''')
            st.markdown(
                '''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📝 현재 기본 매뉴얼인 [**INC14 FC Conveyor & CB Sorter & Spiral 설비 유지보수 Manual**]를 활용하고 있습니다.''')
            st.session_state.query_disabled = False  # 상태 저장용
        elif menu == "📄 운영 매뉴얼 검색":
            st.markdown(
                '''📝 현재 기본 문서인 [**쿠팡 물류센터 운영 매뉴얼**](https://d14ojpq4k4igb1.cloudfront.net/school_edu_guide.pdf)를 활용하고 있습니다.''')
            st.session_state.query_disabled = False  # 상태 저장용
        else:
            st.warning("현재 질의 입력은 비활성화되어 있습니다. 기능은 추후 업데이트될 예정입니다.")
            st.session_state.query_disabled = True  # 상태 저장용


# 'Separately' 옵션 선택 시 나오는 중간 Context를 탭 형태로 보여주는 UI
def show_context_with_tab(contexts):
    tab_category = ["Semantic", "Keyword", "Without Reranker", "Similar Docs"]
    tab_contents = {
        tab_category[0]: [],
        tab_category[1]: [],
        tab_category[2]: [],
        tab_category[3]: []
    }
    for i, contexts_by_doctype in enumerate(contexts):
        tab_contents[tab_category[i]].append(contexts_by_doctype)
    tabs = st.tabs(tab_category)
    for i, tab in enumerate(tabs):
        category = tab_category[i]
        with tab:
            for contexts_by_doctype in tab_contents[category]:
                for context in contexts_by_doctype:
                    st.markdown('##### `정확도`: {}'.format(context["score"]))
                    for line in context["lines"]:
                        st.write(line)
                    parse_metadata(context["meta"])


# 'All at once' 옵션 선택 시 4개의 컬럼으로 나누어 결과 표시하는 UI
def show_answer_with_multi_columns(answers):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('''### `Lexical search` ''')
        st.write(answers[0])
    with col2:
        st.markdown('''### `Semantic search` ''')
        st.write(answers[1])
    with col3:
        st.markdown('''### + `Reranker` ''')
        st.write(answers[2])
    with col4:
        st.markdown('''### + `Parent_docs` ''')
        st.write(answers[3])


####################### Application ###############################
st.set_page_config(layout="wide", page_title="LG CNS 물류센터 챗봇", page_icon="📦")
# Header
st.title(":truck:")
st.title("LG CNS 물류센터 설계 & 운영 챗봇")

st.markdown('''- 이 챗봇은 Amazon Bedrock과 Claude v4 Sonnet 모델로 구현되었습니다.''')
st.markdown('''- 다음과 같은 Advanced RAG 기술을 사용합니다: **Hybrid Search, and Parent Document**''')
st.markdown('''- 원본 데이터는 Amazon OpenSearch에 저장되어 있으며, Amazon Titan 임베딩 모델이 사용되었습니다.''')
st.markdown("""
---
🚀 *해당 챗봇은 LG CNS 물류센터 현장 경험과 내부 문서를 기반으로 구축 중입니다.*
""")
st.markdown('''    ''')

# Store the initial value of widgets in session state
if "document_type" not in st.session_state:
    st.session_state.document_type = "🤖 Chatbot"
if "showing_option" not in st.session_state:
    st.session_state.showing_option = "Separately"
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "Hybrid search"
if "hyde_or_ragfusion" not in st.session_state:
    st.session_state.hyde_or_ragfusion = "None"
disabled = st.session_state.showing_option == "All at once"

with st.sidebar:  # Sidebar 모델 옵션
    st.title("📦 물류센터 챗봇")
    st.markdown("""
        LG CNS 물류센터 관련 정보를 빠르게 찾을 수 있도록 돕는 AI 챗봇입니다. 아래에서 원하는 기능을 선택하세요.
        """)
    # menu = st.radio("기능 선택",
    #         ["🤖 Chatbot", "📄 운영 매뉴얼 검색", "⏏️ 문서 업로드"],
    #         captions=["챗봇이 원하는 조건의 다양한 레퍼런스를 손쉽고 빠르게 찾아줍니다.", "챗봇이 방대한 운영 매뉴얼 문서에서 원하는 정보를 쉽고 빠르게 찾아줍니다.", "원하시는 문서를 직접 업로드해보세요."],
    #         key="document_type",
    # )
    menu = st.radio("기능 선택",
                    ["🤖 Chatbot", "⏏️ 문서 업로드"],
                    captions=["챗봇이 원하는 조건의 다양한 레이아웃과 운영 매뉴얼을 손쉽고 빠르게 찾아줍니다.",
                              "원하시는 문서를 직접 업로드해보세요."],
                    key="document_type",
                    )
    st.markdown("""
        ---
        💡 *PDF, 도면, 매뉴얼 등의 업로드는 추후 버전에서 지원 예정입니다.*
        """)

# Main Interface
if menu == "🤖 Chatbot":
    show_document_info_label()

elif menu == "📄 운영 매뉴얼 검색":
    show_document_info_label()

else:
    show_document_info_label()

###### 'Separately' 옵션 선택한 경우 ######
if st.session_state.showing_option == "Separately":

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요, 무엇이 궁금하세요?"}
        ]
    # 지난 답변 출력
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 유저가 쓴 chat을 query라는 변수에 담음
    if "query_disabled" not in st.session_state or not st.session_state.query_disabled:
        query = st.chat_input("Search documentation")
    else:
        query = None

    if query:
        # Session에 메세지 저장
        st.session_state.messages.append({"role": "user", "content": query})

        # UI에 출력
        with st.chat_message("user"):
            st.markdown(query)

        # Streamlit callback handler로 bedrock streaming 받아오는 컨테이너 설정
        # st_cb = DummyCallback()
        # st_cb = StreamlitCallbackHandler(
        #     st.chat_message("assistant"),
        #     collapse_completed_thoughts=True
        # )
        st_cb = None
        parent = False
        reranker = False
        hyde = False
        ragfusion = False
        # bedrock.py의 invoke 함수 사용
        answer, contexts = glib.invoke(
            query=query,
            streaming_callback=st_cb,
            parent=parent,
            reranker=reranker,
            hyde=hyde,
            ragfusion=ragfusion,
            alpha=False,
            document_type=st.session_state.document_type
        )

        # if hyde or ragfusion:
        #     mid_answer = response[2] if len(response) > 2 else None

        # UI 출력
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for char in answer:
                full_response += char
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # st.chat_message("assistant").write(answer)

        # if hyde:
        #     with st.chat_message("assistant"):
        #         with st.expander("HyDE 중간 생성 답변 ⬇️"):
        #             mid_answer
        # if ragfusion:
        #     with st.chat_message("assistant"):
        #         with st.expander("RAG-Fusion 중간 생성 쿼리 ⬇️"):
        #             mid_answer
        # with st.chat_message("assistant"):
        #     with st.expander("정확도 별 컨텍스트 보기 ⬇️"):
        #         show_context_with_tab(contexts)

        # Session 메세지 저장
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # if hyde or ragfusion:
        #     st.session_state.messages.append({"role": "hyde_or_fusion", "content": mid_answer})
        #
        # st.session_state.messages.append({"role": "assistant_context", "content": contexts})
        # Thinking을 complete로 수동으로 바꾸어 줌
        # st_cb._complete_current_thought()

