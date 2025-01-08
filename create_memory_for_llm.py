from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# step 1 : Load raw PDF(s)
DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
    glob = '*.pdf',
    loader_cls=PyPDFLoader)

documents = loader.load()
return documents





# step 2 : Create Chunks 
# step 3 : Create Vector Embeddings
# step 4 : Store embeddings in FAISS