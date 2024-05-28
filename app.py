import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from GraphRetrieval import GraphRAG, ImageGraphRAG
from PIL import Image
import tempfile

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
        bytes_data = pdf_file.read()
        if st.button("Create Graph from PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(bytes_data)
                tmp_file_path = tmp_file.name

            gr = GraphRAG()
            gr.create_graph_from_pdf(tmp_file_path, similarity_threshold=similarity_threshold)
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


