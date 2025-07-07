import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_folder = os.path.join(base_dir, "localpdf")
    vector_db_path = os.path.join(base_dir, "vectorstore/faiss_index_local_txt")
    load_dotenv()
    if "DASHSCOPE_API_KEY" not in os.environ:
        raise ValueError("环境变量 'DASHSCOPE_API_KEY' 未设置。请检查 .env 文件。")
    embedding_model = DashScopeEmbeddings(
        model='text-embedding-v2',
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )
    txt_files = glob.glob(os.path.join(txt_folder, "*.txt"))
    if not txt_files:
        print(f"未在 {txt_folder} 目录下找到TXT文件。")
        exit(0)
    all_docs = []
    for txt_file in txt_files:
        try:
            loader = TextLoader(txt_file, encoding='utf-8')
            docs = loader.load()
            all_docs.extend(docs)
            print(f"已加载：{txt_file}")
        except Exception as e:
            print(f"加载 {txt_file} 失败: {e}")
    if not all_docs:
        print("未能从TXT中提取任何文本。"); exit(0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    corpus = splitter.split_documents(all_docs)
    vectordb = FAISS.from_documents(corpus, embedding_model)
    vectordb.save_local(vector_db_path)
    print(f"向量化完成，已保存到 {vector_db_path}") 