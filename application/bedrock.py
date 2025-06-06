# bedrock.py
def invoke(query, streaming_callback=None, parent=False, reranker=False, hyde=False, ragfusion=False, alpha=0.5, document_type="Default"):
    # 더미 응답
    answer = f"[모의응답] 질문 '{query}'에 대한 답변입니다."
    contexts = [[{
        "score": round(alpha, 2),
        "lines": [f"관련 문서 문장 1", f"관련 문서 문장 2"],
        "meta": {"category": "Text"}
    }]]
    mid_answer = f"[중간결과] {query}"
    return answer, contexts, mid_answer
