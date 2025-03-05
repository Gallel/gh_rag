import pickle
import faiss
import PyPDF2
import os
from sentence_transformers import SentenceTransformer

class RAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create(self):
        self.sentences = self._load_pdfs("../input/")
        self.index = self._create_index(self.sentences)

    def _load_pdfs(self, directory):
        pages_text = []
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory, filename)
                with open(pdf_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    num_pages = len(reader.pages)
                    
                    # Ometem la primera pàgina de cada PDF
                    pages_text.extend([reader.pages[i].extract_text().strip() for i in range(1, num_pages) if reader.pages[i].extract_text()])
        
        print(f"Loaded {len(pages_text)} pages from {len(os.listdir(directory))} PDFs.")

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

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as file:
            sentences, embeddings = pickle.load(file)
        instance = cls()
        instance.sentences = sentences
        instance.index = faiss.IndexFlatL2(embeddings.shape[1])
        instance.index.add(embeddings)
        return instance

if __name__ == "__main__":
    rag_instance = RAG()
    rag_instance.create()
    output_file = "rag_data.pkl"
    rag_instance.save(output_file)
    print(f"RAG guardat a {output_file}")
