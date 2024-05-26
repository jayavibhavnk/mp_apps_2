import streamlit as st
import os
from GraphRetrieval import GraphRAG, KnowledgeRAG
from langchain_community.graphs import Neo4jGraph
from langchain_text_splitters import CharacterTextSplitter

# Streamlit app
st.title("GraphRetrieval Streamlit App")

# Setting up environment variables
st.header("Set Up Environment Variables")
neo4j_uri = st.text_input("NEO4J_URI", value="add your Neo4j URI here")
neo4j_username = st.text_input("NEO4J_USERNAME", value="add your Neo4j username here")
neo4j_password = st.text_input("NEO4J_PASSWORD", value="add your Neo4j password here", type="password")
openai_api_key = st.text_input("OPENAI_API_KEY", value="add your OpenAI API key here", type="password")

if st.button("Set Environment Variables"):
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USERNAME"] = neo4j_username
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    os.environ['OPENAI_API_KEY'] = openai_api_key
    st.success("Environment variables set!")

# Upload text file to create graph
st.header("Upload Text File to Create Graph")
uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    grag = GraphRAG()
    grag.create_graph_from_file(uploaded_file)
    st.success("Graph created from uploaded file")

    # Query the graph
    st.header("Query the Graph")
    query = st.text_input("Enter your query")

    if st.button("Query"):
        response = grag.queryLLM(query)
        st.write("Response:")
        st.write(response)

        # Switch to greedy search and query
        grag.retrieval_model = "greedy"
        response_greedy = grag.queryLLM(query)
        st.write("Response with Greedy Search:")
        st.write(response_greedy)

# KnowledgeRAG Section
st.header("KnowledgeRAG")

if st.button("Initialize Knowledge Graph"):
    graph = Neo4jGraph()
    gr = KnowledgeRAG()
    gr.init_graph(graph)
    st.success("Knowledge graph initialized")

    # Ingest text into graph
    st.header("Ingest Text Data into Graph")
    text_data = st.text_area("Enter large text data to ingest")

    if st.button("Ingest Data"):
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        docs1 = text_splitter.create_documents([text_data])
        docs = gr.generate_graph_from_text(docs1)
        gr.ingest_data_into_graph(docs)
        gr.init_neo4j_vector_index()
        st.success("Data ingested into knowledge graph")

        # Query the knowledge graph
        st.header("Query the Knowledge Graph")
        knowledge_query = st.text_input("Enter your knowledge graph query")

        if st.button("Query Knowledge Graph"):
            gchain = gr.graphChain()
            response_kg = gchain.invoke({"question": knowledge_query})
            st.write("Knowledge Graph Response:")
            st.write(response_kg)

            # Hybrid search
            gr.hybrid = True
            response_hybrid = gchain.invoke({"question": knowledge_query})
            st.write("Hybrid Search Response:")
            st.write(response_hybrid)

# Image Graph RAG Section
st.header("Image Graph RAG")
image_directory = st.text_input("Enter the directory path for images")

if st.button("Create Image Graph"):
    from GraphRetrieval import ImageGraphRAG

    image_graph_rag = ImageGraphRAG()
    image_paths = image_graph_rag.create_graph_from_directory(image_directory)
    st.success("Image graph created from directory")

    # Search similar images
    uploaded_image = st.file_uploader("Upload an image to search for similar images", type=["jpg", "png"])

    if uploaded_image is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_image.read())
            tmp_file_path = tmp_file.name

        similar_images = image_graph_rag.similarity_search(tmp_file_path, k=5)
        st.header("Similar Images")

        for doc in similar_images:
            st.image(doc.metadata["path"], caption=doc.metadata["path"])

        # Visualize graph
        if st.button("Visualize Image Graph"):
            image_graph_rag.visualize_graph()
            st.success("Image graph visualized")

st.write("#### Contributing")
st.write("Contributions are welcome! Please submit a pull request or open an issue to discuss what you would like to change.")
st.write("#### License")
st.write("This project is licensed under the MIT License. See the LICENSE file for details.")
