from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from src.main_reasoning import reasoning
from src.data_processing.cache_functions import redis_client
from flask_cors import CORS
from src.database.chroma_search_functions import close_chroma_db_connection
import shutil
import time
import os

app = Flask(__name__)
CORS(app)  # This allows all origins

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'test')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PROMPT_TEMPLATE = """
Answer this question in a clear, unboring matter, based on the following context:
{context}
-----
Answer this question based on the above context, without citing the context in your answer:
{question}
Answer:
"""

def stream_response(response_text):
    """Stream the response one character at a time to simulate typing."""
    delay = 0.0001  # Adjust this delay to control the typing speed
    for char in response_text:
        yield char
        time.sleep(delay)

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    response = reasoning(query, PROMPT_TEMPLATE)
    return jsonify({"response": response})

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Here you would typically process the file and add it to Chroma DB
        # You can call your processing function here
        return jsonify({"message": f"File {filename} uploaded successfully to {file_path}"}), 200
    
# Route to clear CV data
@app.route('/clear_cv_data', methods=['POST'])
def clear_cv_data():
    chroma_folder = './data/processed/chroma'

    try:

        close_chroma_db_connection()

        # Check if the chroma folder exists and delete its contents
        if os.path.exists(chroma_folder):
            for filename in os.listdir(chroma_folder):
                file_path = os.path.join(chroma_folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        
        # Clear Redis cache (flush the Redis database)
        redis_client.flushdb()
        return jsonify({"message": "ChromaDB data cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)