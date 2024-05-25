import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from GraphRetrieval import GraphRAG, ImageGraphRAG
from PIL import Image

import os
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import heapq
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI
import pickle
import langchain_core
from queue import PriorityQueue

class GraphDocument(langchain_core.documents.base.Document):
    def __init__(self, page_content, metadata):
        super().__init__(page_content=page_content, metadata=metadata)

    def __repr__(self):
        return f"GraphDocument(page_content='{self.page_content}', metadata={self.metadata})"

class GraphRAG():
    def __init__(self):
        self.graph = None
        self.documents = None
        self.embeddings = None
        self.embedding_model = "all-MiniLM-L6-v2"
        self.retrieval_model = "a_star"

    def constructGraph(self, text, similarity_threshold=0, chunk_size=1250, chunk_overlap=100, metadata=True):
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        pre_documents = text_splitter.create_documents([text])
        documents = [GraphDocument(doc.page_content, doc.metadata) for doc in pre_documents]

        model = SentenceTransformer(self.embedding_model)
        embeddings = model.encode([doc.page_content for doc in documents])
        graph = nx.Graph()

        for i in range(len(documents)):
            for j in range(i, len(documents)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])
                if similarity[0][0] > similarity_threshold:
                    graph.add_edge(i, j, weight=similarity[0][0])

        self.graph = graph
        self.documents = documents
        self.embeddings = embeddings

        return graph, documents, embeddings

    def create_graph_from_file(self, file, similarity_threshold=0):
        with open(file, 'r') as file:
            text_data = file.read()
        self.graph, self.documents, self.embeddings = self.constructGraph(text_data, similarity_threshold=similarity_threshold)
        print("Graph created Successfully!")

    def create_graphs_from_directory(self, directory_path, similarity_threshold=0):
        file_list = []
        overall_text = ""
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(directory_path, file_name)
                temp_text = open(file_path, 'r').read()
                overall_text = overall_text + "\n" + temp_text
                file_list.append((temp_text, file_name))

        self.graph, self.documents, self.embeddings = self.constructGraph(overall_text, similarity_threshold=similarity_threshold)
        print("Graph created Successfully!")

        return file_list

    def create_graph_from_text(self, text, similarity_threshold=0):
        self.graph, self.documents, self.embeddings = self.constructGraph(text, similarity_threshold=similarity_threshold)
        print("Graph created Successfully!")

    def compute_similarity(self, current_node, graph, documents, query_embedding):
        similar_nodes = []
        for neighbor in graph.neighbors(current_node):
            neighbor_embedding = self.embeddings[neighbor]
            neighbor_similarity = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
            similar_nodes.append((neighbor, neighbor_similarity))
        return similar_nodes

    def a_star_search_parallel(self, graph, documents, embeddings, query_text, k=5):
        model = SentenceTransformer(self.embedding_model)
        query_embedding = model.encode([query_text])[0]

        pq = [(0, None, 0)]
        visited = set()
        similar_nodes = []

        while pq and len(similar_nodes) < k:
            _, current_node, similarity_so_far = heapq.heappop(pq)

            if current_node is not None:
                similar_nodes.append((current_node, similarity_so_far))

            results = []
            for neighbor in (graph.neighbors(current_node) if current_node is not None else range(len(documents)-1)):
                results.extend(self.compute_similarity(neighbor, graph, documents, query_embedding))



st.title("Graph Retrieval System")

# Initialize session state
if "input_type" not in st.session_state:
    st.session_state["input_type"] = "Text"
if "gr" not in st.session_state:
    st.session_state["gr"] = None
if "igr" not in st.session_state:
    st.session_state["igr"] = None

# Sidebar for selecting input data type
st.session_state["input_type"] = st.sidebar.selectbox("Select Input Data Type", ["Text", "PDF", "Images"], index=["Text", "PDF", "Images"].index(st.session_state["input_type"]))

if st.session_state["input_type"] == "Text":
    text_input = st.text_area("Enter Text", height=200)
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05)

    if st.button("Create Graph from Text"):
        gr = GraphRAG()
        gr.create_graph_from_text(text_input, similarity_threshold=similarity_threshold)
        st.session_state["gr"] = gr
        st.success("Graph created successfully!")

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(gr.graph)
        nx.draw(gr.graph, pos, with_labels=True, ax=ax)
        st.pyplot(fig)

    query = st.text_input("Enter Query")
    if st.button("Query Graph"):
        if st.session_state["gr"] is not None:
            result = st.session_state["gr"].queryLLM(query)
            st.write(result)
        else:
            st.warning("Please create a graph first.")

elif st.session_state["input_type"] == "PDF":
    pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05)

    if pdf_file is not None:
        if st.button("Create Graph from PDF"):
            gr = GraphRAG()
            gr.create_graph_from_pdf(pdf_file, similarity_threshold=similarity_threshold)
            st.session_state["gr"] = gr
            st.success("Graph created successfully!")

            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(gr.graph)
            nx.draw(gr.graph, pos, with_labels=True, ax=ax)
            st.pyplot(fig)

    query = st.text_input("Enter Query")
    if st.button("Query Graph"):
        if st.session_state["gr"] is not None:
            result = st.session_state["gr"].queryLLM(query)
            st.write(result)
        else:
            st.warning("Please create a graph first.")

elif st.session_state["input_type"] == "Images":
    uploaded_images = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05)

    if st.button("Create Graph from Images"):
        if uploaded_images:
            image_paths = [Image.open(image) for image in uploaded_images]
            igr = ImageGraphRAG()
            igr.constructGraph(image_paths, similarity_threshold=similarity_threshold)
            st.session_state["igr"] = igr
            st.success("Graph created successfully!")

            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(igr.graph)
            nx.draw(igr.graph, pos, with_labels=True, ax=ax)
            st.pyplot(fig)

    query_image = st.file_uploader("Upload Query Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if st.button("Query Graph"):
        if st.session_state["igr"] is not None and query_image is not None:
            query_image_path = query_image.name
            result = st.session_state["igr"].similarity_search(query_image_path, k=5)
            st.write(f"Top 5 similar images for the query image '{query_image_path}':")
            for doc in result:
                st.image(doc.page_content, caption=doc.page_content, use_column_width=True)
        else:
            st.warning("Please create a graph and upload a query image.")
