import os, sys, boto3
from utils.rag_summit import prompt_repo, OpenSearchHybridSearchRetriever, prompt_repo, qa_chain
from utils.opensearch_summit import opensearch_utils
from utils.ssm import parameter_store
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from utils import bedrock
from utils.bedrock import bedrock_info
from langchain_aws.agent_rag import BedrockAgent

region = boto3.Session().region_name
pm = parameter_store(region)
secrets_manager = boto3.client('secretsmanager', region_name=region)


def get_agent():
    agent_id = os.environ.get("BEDROCK_AGENT_ID")
    alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "DRAFT")
    region = os.environ.get("AWS_DEFAULT_REGION")
    assume_role = os.environ.get("BEDROCK_ASSUME_ROLE")

    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=assume_role,
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=region,
    )

    return BedrockAgent(
        client=boto3_bedrock,
        agent_id=agent_id,
        agent_alias_id=alias_id,
        region=region,
    )


def invoke(query, streaming_callback, parent, reranker, hyde, ragfusion, alpha, document_type="Default"):
    use_agent = os.environ.get("USE_BEDROCK_AGENT", "false").lower() == "true"

    if use_agent:
        agent = get_agent()
        print("Using Bedrock Agent...")
        response = agent.invoke(input=query)
        return response["output"] if isinstance(response, dict) and "output" in response else response, []

    # 기존 방식 유지
    llm_text = get_llm(streaming_callback)
    opensearch_hybrid_retriever = get_retriever(streaming_callback, parent, reranker, hyde, ragfusion, alpha,
                                                document_type)
    system_prompt = prompt_repo.get_system_prompt()

    qa = qa_chain(
        llm_text=llm_text,
        retriever=opensearch_hybrid_retriever,
        system_prompt=system_prompt,
        return_context=False,
        verbose=False
    )

    response, pretty_contexts, similar_docs, augmentation = qa.invoke(query=query, complex_doc=True)

    if hyde or ragfusion:
        return response, pretty_contexts, augmentation

    if not hyde or ragfusion:
        if alpha == 0.0:
            pretty_contexts[0].clear()
        elif alpha == 1.0:
            pretty_contexts[1].clear()

    return response, pretty_contexts


def get_llm(streaming_callback):
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )
    llm = ChatBedrock(
        model_id=bedrock_info.get_model_id(model_name="Claude-V3-3.5-Sonne"),
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
