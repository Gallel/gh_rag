import pickle
import faiss
import PyPDF2
import os
from sentence_transformers import SentenceTransformer

class RAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create(self, pdf_path):
        self.sentences = self._load_pdf(pdf_path)
        self.index = self._create_index(self.sentences)

    def _load_pdf(self, pdf_path):
        pages_text = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            # Ometem la primera pàgina del PDF
            pages_text.extend([reader.pages[i].extract_text().strip() for i in range(1, num_pages) if reader.pages[i].extract_text()])
        
        print(f"Loaded {len(pages_text)} pages from {pdf_path}.")
        return pages_text

    def _create_index(self, sentences):
        embeddings = self.model.encode(sentences)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, prompt):
        prompt_embedding = self.model.encode([prompt])
        D, I = self.index.search(prompt_embedding, k=1)
        return [self.sentences[i] for i in I[0]] if I[0][0] != -1 else ["No relevant sentence found."]

    def save(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump((self.sentences, self.index.reconstruct_n(0, len(self.sentences))), file)

if __name__ == "__main__":
    input_directory = "../input/"
    pdf_files = [f for f in os.listdir(input_directory) if f.endswith(".pdf")]
    
    for idx, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(input_directory, pdf_file)
        rag_instance = RAG()
        rag_instance.create(pdf_path)
        output_file = f"../model/rag_data_{idx}.pkl"
        rag_instance.save(output_file)
        print(f"RAG guardat a {output_file}")
