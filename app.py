import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from GraphRetrieval import GraphRAG, ImageGraphRAG
from PIL import Image

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
            st.write(st.session_state['gr'])
            result = gr.queryLLM(query)
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
