# Auto ML with GCP and PyCaret
This repository contains three folders with different projects that can be run independently of each other:
- **gcp_vertex_auto_ml**: contains a Google Cloud Platform Vertex AI notebook for working with Auto ML tabular models using the SDK. This pipeline includes training, evaluation, and batch prediction.
- **pycaret**: contains a PyCaret experiment notebook for creating a regression model using Auto ML and custom settings.
- **streamlit**: contains a streamlit protype for PyCaret modeling. To run this app you need to follow the next steps:
  - Install all dependecies using `pip install -r requirements.txt`
  - Run the app `streamlit run app.py`
