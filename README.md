# RAG System

This project implements a Retrieval-Augmented Generation (RAG) system that retrieves relevant sentences based on user input prompts.

## Project Structure

```
rag-system
├── src
│   ├── main.py        # Entry point of the application
│   └── rag.py         # Contains the RAG class for sentence retrieval
├── requirements.txt    # Lists the dependencies required for the project
└── README.md           # Documentation for the project
```

## Setup Instructions

1. **Install Dependencies**  
   Navigate to the project directory and install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

2. **Run the Application**  
   Execute the main application by running the following command:

   ```
   python src/main.py
   ```

3. **Input a Prompt**  
   When prompted, enter a sentence or keyword. The system will return the most relevant sentence from the predefined set of sentences.

## Example Sentences

The system is initialized with the following sentences:
1. "The sky is blue."
2. "Cats are great companions."
3. "Python is a versatile programming language."
4. "The sun rises in the east."
5. "Reading books expands your knowledge."

Feel free to modify the sentences in `src/rag.py` to customize the RAG system's responses.