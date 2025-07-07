import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.embeddings import DashScopeEmbeddings 

class LocalRAGAgent:
    def __init__(self, local_txt_folder=None, vector_db_path=None):
        # 默认路径修正为相对于本文件的绝对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if local_txt_folder is None:
            local_txt_folder = os.path.join(base_dir, "../my_documents")
        if vector_db_path is None:
            vector_db_path = os.path.join(base_dir, "../faiss_index_local_txts")
        load_dotenv()
        if "DEEPSEEK_API_KEY" not in os.environ:
            raise ValueError("环境变量 'DEEPSEEK_API_KEY' 未设置。请检查 .env 文件。")
        if "DASHSCOPE_API_KEY" not in os.environ:
            raise ValueError("环境变量 'DASHSCOPE_API_KEY' 未设置。请检查 .env 文件。")
        self.llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
        try:
            self.embedding_model = DashScopeEmbeddings(
                model='text-embedding-v2',
                dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
            )
        except Exception as e:
            raise RuntimeError(f"DashScope Embeddings 模型初始化失败: {e}")
        try:
            reranker_model_instance = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
            self.reranker = CrossEncoderReranker(model=reranker_model_instance, top_n=5)
        except Exception as e:
            raise RuntimeError(f"Rerank 模型初始化失败: {e}")
        if not os.path.exists(local_txt_folder):
            raise FileNotFoundError(f"本地文件夹 '{local_txt_folder}' 不存在。")
        all_files_in_folder = [os.path.join(local_txt_folder, f) for f in os.listdir(local_txt_folder) if f.endswith(".txt")]
        vectordb = None
        if os.path.exists(vector_db_path):
            try:
                vectordb = FAISS.load_local(vector_db_path, self.embedding_model, allow_dangerous_deserialization=True)
            except Exception:
                vectordb = None
        documents_to_index = []
        for file_path in all_files_in_folder:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents_to_index.extend(loader.load())
            except Exception:
                pass
        if not documents_to_index:
            raise RuntimeError(f"在文件夹 '{local_txt_folder}' 中未找到任何 TXT 文件或加载失败。")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        corpus = splitter.split_documents(documents_to_index)
        if vectordb is None or len(corpus) != (vectordb.index.ntotal if hasattr(vectordb.index, 'ntotal') else -1):
            vectordb = FAISS.from_documents(corpus, self.embedding_model)
        vectordb.save_local(vector_db_path)
        base_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        self.retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=self.reranker
        )
        template = """您是一个问答聊天机器人。\n请仅使用给定的上下文回答问题。如果信息不足，请说明。\n<context>{context}</context>\n问题: {input}"""
        prompt = ChatPromptTemplate.from_template(template)
        self.doc_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, self.doc_chain)

    def ask(self, question):
        response = self.chain.invoke({"input": question})
        answer = response.get('answer', '')
        context = response.get('context', [])
        return answer, context 