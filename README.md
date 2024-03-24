### Guide to Understanding the Code for Creating a Vectordatabase 

I'll walk you through the code step by step to explain what a vectordatabase is and how it's being created using Python and various libraries.

### Prerequisites:
1. Install [Ollama](https://ollama.com/) 🦙
2. Run Ollama in the back:
```bash
ollama serve
```
3. Virtual Environment
```bash
Make venv
```

3. Install requirements
```bash
Make install
```

1. **Importing Necessary Libraries:**
   ```python
   import langchain
   import pandas
   from sentence_transformers import SentenceTransformer, util
   from utils import *
   from langchain.text_splitter import TokenTextSplitter
   import chromadb
   import ollama
   ```

   Here, we import libraries such as langchain, pandas, SentenceTransformer, utils, langchain.text_splitter, chromadb, and ollama. These libraries are used for text processing, embedding sentences, managing databases, and conducting language-based tasks.

2. **Reading and Preprocessing Text Data:**
   ```python
   with open("resume.txt", "r") as file:
       text_to_split = file.read()
   text_splitter = TokenTextSplitter(encoding_name="gpt2", chunk_size=250, chunk_overlap=10)
   texts = text_splitter.split_text(text_to_split)
   texts = list(map(lambda x: x.replace("\n", " "), texts))
   ```

   This section reads text data from a file ("resume.txt"), splits it into chunks using TokenTextSplitter, and preprocesses it by removing newline characters.

3. **Generating Text Embeddings:**
   ```python
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   embedding_list = model.encode(texts)
   ```

   Using SentenceTransformer, we encode the preprocessed text chunks into embeddings. These embeddings represent the semantic meaning of each text chunk.

4. **Working with ChromaDB:**
   ```python
   chroma_client = chromadb.Client()
   collection_name = "mycollection"
   collections = chroma_client.list_collections()
   collection_names = [collection.name for collection in collections]
   ```

   We initialize a ChromaDB client and interact with a collection named "mycollection" within the database. We also retrieve existing collection names to check if "mycollection" already exists.

5. **Creating or Loading Collection in ChromaDB:**
   ```python
   if collection_name in collection_names:
       collection = chroma_client.get_collection(collection_name)
       print("Loaded existing collection:", collection_name)
   else:
       collection = chroma_client.create_collection(name=collection_name)
       print("Created new collection:", collection_name)
   ```

   Depending on whether "mycollection" exists, we either load the existing collection or create a new one.

6. **Adding Data to Collection:**
   ```python
   collection.add(
       embeddings=embedding_list,
       documents=texts,
       ids=["ID" + str(i) for i in range(len(embedding_list))],
   )
   ```

   We add the text embeddings, corresponding documents, and unique IDs to the collection in ChromaDB. This step populates the vectordatabase with our encoded text data.

7. **Querying the Collection:**
   ```python
   question = "Tell me about Rafael Davila"
   similar_documents = collection.query(query_texts=[question], n_results=3)
   ```

   We query the collection to find similar documents based on a given query text ("Tell me about sodemca") and retrieve the top 3 most similar documents.

8. **Using Ollama for Language-Based Tasks:**
   ```python
   content = f"Answer the following question:{question}\
       With this context:{context}.\
       Do not provide answers outside this context."

   response = ollama.chat(
       model="llama2",
       messages=[
           {"role": "user", "content": content},
       ],
   )
   print(response["message"]["content"])
   ```

   Finally, we use Ollama to engage in a conversation based on a given question within a specific context. The response contains the answer provided by the model within the defined context.

   This is the output:

```bash
Rafael Davila Bugarín is a Mexican data scientist who has received the Fulbright-García Robles scholarship to pursue his second master's degree in Data Science at Duke University in North Carolina, USA. He works at the Instituto Federal de Telecomunicaciones and values the role of mathematics in his career. Rafael is also passionate about inspiring young minds through mathematics clubs for children.

Rafael's technical skills include Python, PyTorch, Tensorflow, PySpark, R, SQL, Tableau, Power BI, git, emacs, docker, and AWS, among others. He has relevant experience as a research assistant at the Duke Center for Research & Engineering of AI Technology in Education, where he develops AI-based software tools to enhance teaching and learning.

Rafael is also involved in various educational initiatives, including the Mathematics Club and Mathematics Olympiads, which aim to promote mathematics education in Mexico. He is part of the SODEMCA Overview, a summer course that encourages critical thinking through playful activities, strategy games, logic games, and a pre-algebra showcase.

According to Rafael, learning mathematics can have numerous benefits for children, including improving their mental abilities, problem-solving skills, communication, collaboration, and creativity. He believes that by providing evidence-based support to parents and incorporating innovative educational materials, he can help nurture mathletes and contribute to the intellectual growth of Mexico.
```

This code essentially demonstrates the process of creating and utilizing a vectordatabase to store and retrieve text data efficiently for language-related tasks.