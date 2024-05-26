import streamlit as st
import os
from GraphRetrieval import GraphRAG, KnowledgeRAG, ImageGraphRAG
from langchain_community.graphs import Neo4jGraph
from langchain_text_splitters import CharacterTextSplitter

# Streamlit app
st.title("GraphRetrieval Streamlit App")

# Sidebar for environment variables
st.sidebar.header("Set Up Environment Variables")
neo4j_uri = st.sidebar.text_input("NEO4J_URI", value="add your Neo4j URI here")
neo4j_username = st.sidebar.text_input("NEO4J_USERNAME", value="add your Neo4j username here")
neo4j_password = st.sidebar.text_input("NEO4J_PASSWORD", value="add your Neo4j password here", type="password")

if st.sidebar.button("Set Environment Variables"):
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USERNAME"] = neo4j_username
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    st.sidebar.success("Environment variables set!")

# Choose RAG type
st.header("Choose RAG Type")
rag_type = st.selectbox("Select RAG Type", ["GraphRAG", "KnowledgeRAG", "ImageRAG"])

if rag_type == "GraphRAG":
    st.header("GraphRAG")

    # Upload text file to create graph
    uploaded_file = st.file_uploader("Choose a text file", type="txt")

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        grag = GraphRAG()
        grag.create_graph_from_file(uploaded_file)
        st.success("Graph created from uploaded file")

        # Query the graph
        st.header("Query the Graph")
        query = st.text_input("Enter your query")

        retrieval_model = st.selectbox("Select Retrieval Model", ["default", "greedy"])

        if st.button("Query"):
            if retrieval_model == "greedy":
                grag.retrieval_model = "greedy"
            response = grag.queryLLM(query)
            st.write("Response:")
            st.write(response)

elif rag_type == "KnowledgeRAG":
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

        search_type = st.selectbox("Select Search Type", ["Regular Search", "Hybrid Search"])

        if st.button("Query Knowledge Graph"):
            gchain = gr.graphChain()
            if search_type == "Hybrid Search":
                gr.hybrid = True
            else:
                gr.hybrid = False
            response_kg = gchain.invoke({"question": knowledge_query})
            st.write("Knowledge Graph Response:")
            st.write(response_kg)

elif rag_type == "ImageRAG":
    # Image Graph RAG Section
    st.header("Image Graph RAG")
    image_directory = st.text_input("Enter the directory path for images")
    
    if st.button("Create Image Graph"):
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