import os, sys, boto3
from utils.rag_summit import prompt_repo, OpenSearchHybridSearchRetriever, prompt_repo, qa_chain
from utils.opensearch_summit import opensearch_utils
from utils.ssm import parameter_store
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from utils import bedrock
from utils.bedrock import bedrock_info

region = boto3.Session().region_name
pm = parameter_store(region)
secrets_manager = boto3.client('secretsmanager', region_name=region)


def invoke_agent_direct(query):
    agent_id = os.environ.get("BEDROCK_AGENT_ID")
    alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "DRAFT")
    session_id = os.environ.get("SESSION_ID", "default-session")
    region = os.environ.get("AWS_DEFAULT_REGION")

    client = boto3.client("bedrock-agent-runtime", region_name=region)

    # 사용자 포맷에 맞게 request_body 생성
    request_body = {
        "requestBody": {
            "content": {
                "application/json": {
                    "properties": [
                        {
                            "name": "query",
                            "value": query
                        }
                    ]
                }
            }
        }
    }

    response = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        inputText=json.dumps(request_body)
    )

    # Check if streaming response is returned
    if "completion" in response:
        output = b""
        for event in response["completion"]:
            chunk = event.get("chunk", {}).get("bytes")
            if chunk:
                output += chunk
        return output.decode("utf-8"), []
    else:
        return response.get("outputText", ""), []


# def invoke(query, streaming_callback, parent, reranker, hyde, ragfusion, alpha, document_type="Default"):
#     use_agent = os.environ.get("USE_BEDROCK_AGENT", "false").lower() == "true"
#
#     if use_agent:
#         print("Using Bedrock Agent directly...")
#         return invoke_agent_direct(query)
#
#     # 기존 방식 유지
#     llm_text = get_llm(streaming_callback)
#     opensearch_hybrid_retriever = get_retriever(streaming_callback, parent, reranker, hyde, ragfusion, alpha, document_type)
#     system_prompt = prompt_repo.get_system_prompt()
#
#     qa = qa_chain(
#         llm_text=llm_text,
#         retriever=opensearch_hybrid_retriever,
#         system_prompt=system_prompt,
#         return_context=False,
#         verbose=False
#     )
#
#     response, pretty_contexts, similar_docs, augmentation = qa.invoke(query=query, complex_doc=True)
#
#     if hyde or ragfusion:
#         return response, pretty_contexts, augmentation
#
#     if not hyde or ragfusion:
#         if alpha == 0.0:
#             pretty_contexts[0].clear()
#         elif alpha == 1.0:
#             pretty_contexts[1].clear()
#
#     return response, pretty_contexts

def invoke(query, streaming_callback=None, parent=None, reranker=None, hyde=None, ragfusion=None, alpha=0.5, document_type="Default"):
    # 사용자 정의 Bedrock Agent만 사용하여 호출
    response, _ = invoke_agent_direct(query)
    return response, []

def get_llm(streaming_callback):
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )
    llm = ChatBedrock(
        model_id=bedrock_info.get_model_id(model_name="Claude-V3.5-Sonnet"),
        client=boto3_bedrock,
        model_kwargs={
            "max_tokens": 1024,
            "stop_sequences": ["\n\nHuman"],
        },
        streaming=True,
        callbacks=[streaming_callback],
    )
    return llm


def get_embedding_model(document_type):
    model_id = 'amazon.titan-embed-text-v1' if document_type == 'Default' else 'amazon.titan-embed-text-v2:0'
    llm_emb = BedrockEmbeddings(model_id=model_id)
    return llm_emb


def get_opensearch_client():
    opensearch_domain_endpoint = pm.get_params(key='opensearch_alllay_domain_endpoint', enc=False)
    opensearch_user_id = pm.get_params(key='opensearch_user_id_alllay', enc=False)
    response = secrets_manager.get_secret_value(SecretId='opensearch_user_password_alllay')
    secrets_string = response.get('SecretString')
    secrets_dict = eval(secrets_string)
    opensearch_user_password = secrets_dict['pwkey']

    http_auth = (opensearch_user_id, opensearch_user_password)
    aws_region = os.environ.get("AWS_DEFAULT_REGION", None)

    return opensearch_utils.create_aws_opensearch_client(
        aws_region,
        opensearch_domain_endpoint,
        http_auth
    )


def get_retriever(streaming_callback, parent, reranker, hyde, ragfusion, alpha, document_type):
    os_client = get_opensearch_client()
    llm_text = get_llm(streaming_callback)
    llm_emb = get_embedding_model(document_type)
    reranker_endpoint_name = "reranker-alllay"
    index_name = "default_doc_index" if document_type == "Default" else "customer_doc_index"

    return OpenSearchHybridSearchRetriever(
        os_client=os_client,
        index_name=index_name,
        llm_text=llm_text,
        llm_emb=llm_emb,
        minimum_should_match=0,
        filter=[],
        fusion_algorithm="RRF",
        complex_doc=True,
        ensemble_weights=[alpha, 1.0 - alpha],
        reranker=reranker,
        reranker_endpoint_name=reranker_endpoint_name,
        parent_document=parent,
        rag_fusion=ragfusion,
        rag_fusion_prompt=prompt_repo.get_rag_fusion(),
        hyde=hyde,
        hyde_query=['web_search'],
        query_augmentation_size=3,
        async_mode=True,
        k=7,
        verbose=True,
    )

def hybrid_search(streaming_callback, query: str):
    index_name = "alllay_index"
    opensearch_domain_endpoint = pm.get_params(
        key="opensearch_domain_endpoint",
        enc=False
    )


    secrets_manager = boto3.client('secretsmanager')

    response = secrets_manager.get_secret_value(
        SecretId='opensearch_user_password'
    )

    secrets_string = response.get('SecretString')
    secrets_dict = eval(secrets_string)

    opensearch_user_id = secrets_dict['es.net.http.auth.user']
    opensearch_user_password = secrets_dict['pwkey']

    opensearch_domain_endpoint = opensearch_domain_endpoint
    rag_user_name = opensearch_user_id
    rag_user_password = opensearch_user_password

    http_auth = (rag_user_name, rag_user_password) # Master username, Master password

    aws_region = os.environ.get("AWS_DEFAULT_REGION", None)

    os_client = opensearch_utils.create_aws_opensearch_client(
        aws_region,
        opensearch_domain_endpoint,
        http_auth
    )
    opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
        os_client=os_client,
        index_name=index_name,
        llm_text=get_llm(streaming_callback), # llm for query augmentation in both rag_fusion and HyDE
        llm_emb=get_embedding_model(), # Used in semantic search based on opensearch

        # hybird-search debugger
        #hybrid_search_debugger = "semantic", #[semantic, lexical, None]

        # option for lexical
        minimum_should_match=0,
        filter=[],

        # option for search
        fusion_algorithm="RRF", # ["RRF", "simple_weighted"], rank fusion 방식 정의
        ensemble_weights=[.51, .49], # [for semantic, for lexical], Semantic, Lexical search 결과에 대한 최종 반영 비율 정의
        reranker=False, # enable reranker with reranker model
        #reranker_endpoint_name=endpoint_name, # endpoint name for reranking model
        parent_document=True, # enable parent document

        # option for complex pdf consisting of text, table and image
        complex_doc=True,

        # option for async search
        async_mode=True,

        # option for output
        k=7, # 최종 Document 수 정의
        verbose=False,
    )
    search_hybrid_result, tables, images = opensearch_hybrid_retriever.get_relevant_documents(query)
    result = ""
    for idx, context in enumerate(search_hybrid_result):
        result += f"Document {idx+1}:"
        result += f"Page Content: {context.page_content}"
        result += f"Metadata: {context.metadata}"
        result += "="*50
    return result
