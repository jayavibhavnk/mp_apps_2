import streamlit as st
import os
from GraphRetrieval import GraphRAG, KnowledgeRAG, ImageGraphRAG
from langchain_community.graphs import Neo4jGraph
from langchain_text_splitters import CharacterTextSplitter
from py2neo import Graph
import networkx as nx
from pyvis.network import Network
import tempfile

# Initialize session state variables
if 'neo4j_uri' not in st.session_state:
    st.session_state['neo4j_uri'] = "add your Neo4j URI here"
if 'neo4j_username' not in st.session_state:
    st.session_state['neo4j_username'] = "add your Neo4j username here"
if 'neo4j_password' not in st.session_state:
    st.session_state['neo4j_password'] = "add your Neo4j password here"
if 'graph_initialized' not in st.session_state:
    st.session_state['graph_initialized'] = False
if 'data_ingested' not in st.session_state:
    st.session_state['data_ingested'] = False
if 'graph_created' not in st.session_state:
    st.session_state['graph_created'] = False
if 'grag' not in st.session_state:
    st.session_state['grag'] = None
if 'gr' not in st.session_state:
    st.session_state['gr'] = None
if 'image_graph_rag' not in st.session_state:
    st.session_state['image_graph_rag'] = None

# Streamlit app
st.title("GraphRetrieval Streamlit App")

# Sidebar for environment variables
st.sidebar.header("Set Up Environment Variables")
st.session_state['neo4j_uri'] = st.sidebar.text_input("NEO4J_URI", value=st.session_state['neo4j_uri'])
st.session_state['neo4j_username'] = st.sidebar.text_input("NEO4J_USERNAME", value=st.session_state['neo4j_username'])
st.session_state['neo4j_password'] = st.sidebar.text_input("NEO4J_PASSWORD", value=st.session_state['neo4j_password'], type="password")

if st.sidebar.button("Set Environment Variables"):
    os.environ["NEO4J_URI"] = st.session_state['neo4j_uri']
    os.environ["NEO4J_USERNAME"] = st.session_state['neo4j_username']
    os.environ["NEO4J_PASSWORD"] = st.session_state['neo4j_password']
    st.sidebar.success("Environment variables set!")

# Choose RAG type
st.header("Choose RAG Type")
rag_type = st.selectbox("Select RAG Type", ["GraphRAG", "KnowledgeRAG", "ImageRAG"])

if rag_type == "GraphRAG":
    st.header("GraphRAG")

    # Upload text file to create graph
    uploaded_file = st.file_uploader("Choose a text file", type="txt")

    if uploaded_file is not None and not st.session_state['graph_created']:
        text = uploaded_file.read().decode("utf-8")
        st.session_state['grag'] = GraphRAG()
        st.session_state['grag'].create_graph_from_file(uploaded_file)
        st.session_state['graph_created'] = True
        st.success("Graph created from uploaded file")

    # Query the graph
    if st.session_state['graph_created']:
        st.header("Query the Graph")
        query = st.text_input("Enter your query")

        retrieval_model = st.selectbox("Select Retrieval Model", ["default", "greedy"])

        if st.button("Query"):
            if retrieval_model == "greedy":
                st.session_state['grag'].retrieval_model = "greedy"
            response = st.session_state['grag'].queryLLM(query)
            st.write("Response:")
            st.write(response)
        
        # Visualize graph
        if st.button("Visualize Graph"):
            st.header("Graph Visualization")
            graph = Graph(st.session_state['neo4j_uri'], auth=(st.session_state['neo4j_username'], st.session_state['neo4j_password']))
            nodes_query = "MATCH (n) RETURN n LIMIT 100"
            relationships_query = "MATCH ()-[r]->() RETURN r LIMIT 100"
            
            nodes = graph.run(nodes_query).data()
            relationships = graph.run(relationships_query).data()
            
            G = nx.Graph()
            
            for node in nodes:
                node = node['n']
                G.add_node(node.identity, label=node.labels[0], title=node['name'] if 'name' in node else "")
                
            for rel in relationships:
                rel = rel['r']
                G.add_edge(rel.start_node.identity, rel.end_node.identity, title=rel.type)
                
            net = Network(notebook=True)
            net.from_nx(G)
            
            path = tempfile.mktemp(suffix=".html")
            net.show(path)
            with open(path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=600)

elif rag_type == "KnowledgeRAG":
    st.header("KnowledgeRAG")

    if st.button("Initialize Knowledge Graph") and not st.session_state['graph_initialized']:
        graph = Neo4jGraph()
        st.session_state['gr'] = KnowledgeRAG()
        st.session_state['gr'].init_graph(graph)
        st.session_state['graph_initialized'] = True
        st.success("Knowledge graph initialized")

    # Ingest text into graph
    if st.session_state['graph_initialized']:
        st.header("Ingest Text Data into Graph")
        text_data = st.text_area("Enter large text data to ingest")

        if st.button("Ingest Data") and not st.session_state['data_ingested']:
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            docs1 = text_splitter.create_documents([text_data])
            docs = st.session_state['gr'].generate_graph_from_text(docs1)
            st.session_state['gr'].ingest_data_into_graph(docs)
            st.session_state['gr'].init_neo4j_vector_index()
            st.session_state['data_ingested'] = True
            st.success("Data ingested into knowledge graph")

    # Query the knowledge graph
    if st.session_state['data_ingested']:
        st.header("Query the Knowledge Graph")
        knowledge_query = st.text_input("Enter your knowledge graph query")

        search_type = st.selectbox("Select Search Type", ["Regular Search", "Hybrid Search"])

        if st.button("Query Knowledge Graph"):
            gchain = st.session_state['gr'].graphChain()
            st.session_state['gr'].hybrid = (search_type == "Hybrid Search")
            response_kg = gchain.invoke({"question": knowledge_query})
            st.write("Knowledge Graph Response:")
            st.write(response_kg)
        
        # Visualize graph
        if st.button("Visualize Graph"):
            st.header("Graph Visualization")
            graph = Graph(st.session_state['neo4j_uri'], auth=(st.session_state['neo4j_username'], st.session_state['neo4j_password']))
            nodes_query = "MATCH (n) RETURN n LIMIT 100"
            relationships_query = "MATCH ()-[r]->() RETURN r LIMIT 100"
            
            nodes = graph.run(nodes_query).data()
            relationships = graph.run(relationships_query).data()
            
            G = nx.Graph()
            
            for node in nodes:
                node = node['n']
                G.add_node(node.identity, label=node.labels[0], title=node['name'] if 'name' in node else "")
                
            for rel in relationships:
                rel = rel['r']
                G.add_edge(rel.start_node.identity, rel.end_node.identity, title=rel.type)
                
            net = Network(notebook=True)
            net.from_nx(G)
            
            path = tempfile.mktemp(suffix=".html")
            net.show(path)
            with open(path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=600)

elif rag_type == "ImageRAG":
    # Image Graph RAG Section
    st.header("Image Graph RAG")
    image_directory = st.text_input("Enter the directory path for images")
    
    if st.button("Create Image Graph"):
        st.session_state['image_graph_rag'] = ImageGraphRAG()
        image_paths = st.session_state['image_graph_rag'].create_graph_from_directory(image_directory)
        st.session_state['graph_created'] = True
        st.success("Image graph created from directory")
    
    if st.session_state['graph_created']:
        # Search similar images
        uploaded_image = st.file_uploader("Upload an image to search for similar images", type=["jpg", "png"])
    
        if uploaded_image is not None:
            import tempfile
    
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_image.read())
                tmp_file_path = tmp_file.name
    
            similar_images = st.session_state['image_graph_rag'].similarity_search(tmp_file_path, k=5)
            st.header("Similar Images")
    
            for doc in similar_images:
                st.image(doc.metadata["path"], caption=doc.metadata["path"])
    
            # Visualize graph
            if st.button("Visualize Image Graph"):
                st.session_state['image_graph_rag'].visualize_graph()
                st.success("Image graph visualized")
