from flask import Flask, render_template, render_template_string, request
import os
import sqlite3
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


app = Flask(__name__)


# =========================================================
# SETTINGS
# =========================================================

UPLOAD_FOLDER = "static/uploads"
DATABASE = "lost_found.db"
MODEL_PATH = "model/lost_item_model.keras"
MODEL_PATH = "model/lost_item_model.h5"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# LOAD CUSTOM AI MODEL
# =========================================================

print("Loading Custom AI Model...")

model = load_model(MODEL_PATH)

print("Custom AI Model Loaded Successfully!")


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "bag",
    "key",
    "pen",
    "pencil",
    "phone"
]


# =========================================================
# DATABASE
# =========================================================

def create_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    # LOST ITEMS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT,
            description TEXT,
            lost_date TEXT,
            location TEXT,
            contact TEXT,
            image_name TEXT,
            prediction TEXT,
            confidence REAL
        )
    """)

    # FOUND ITEMS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT,
            description TEXT,
            found_date TEXT,
            location TEXT,
            contact TEXT,
            image_name TEXT,
            prediction TEXT,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()


create_database()


# =========================================================
# AI PREDICTION FUNCTION
# =========================================================

def predict_image(image_path):

    print("Analyzing image...")

    img = keras_image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = keras_image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Same preprocessing used during training
    img_array = preprocess_input(img_array)

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    prediction = class_names[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    ) * 100

    print("Prediction:", prediction)
    print("Confidence:", round(confidence, 2), "%")

    return prediction, confidence


# =========================================================
# IMAGE FEATURE EXTRACTION
# =========================================================

def extract_features(image_path):

    img = keras_image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = keras_image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    # Custom MobileNetV2 base model
    base_model = model.layers[0]

    features = base_model.predict(
        img_array,
        verbose=0
    )

    # Average spatial features
    features = np.mean(
        features,
        axis=(1, 2)
    )[0]

    # Normalize feature vector
    norm = np.linalg.norm(features)

    if norm != 0:

        features = features / norm

    return features


# =========================================================
# IMAGE SIMILARITY
# =========================================================

def calculate_similarity(image1, image2):

    feature1 = extract_features(image1)

    feature2 = extract_features(image2)

    # Cosine similarity
    similarity = np.dot(
        feature1,
        feature2
    )

    # Convert to percentage
    similarity = float(similarity) * 100

    similarity = max(
        0,
        min(100, similarity)
    )

    return similarity


# =========================================================
# FIND BEST LOST ITEM MATCH
# =========================================================

def find_best_match(found_image_path, found_prediction):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            item_name,
            description,
            lost_date,
            location,
            contact,
            image_name,
            prediction,
            confidence
        FROM items
    """)

    lost_items = cursor.fetchall()

    conn.close()


    if not lost_items:

        return None


    best_match = None

    best_score = 0


    for item in lost_items:

        (
            item_id,
            item_name,
            description,
            lost_date,
            location,
            contact,
            image_name,
            prediction,
            confidence
        ) = item


        lost_image_path = os.path.join(
            UPLOAD_FOLDER,
            image_name
        )


        # Skip if image no longer exists
        if not os.path.exists(lost_image_path):

            continue


        # Image similarity
        image_score = calculate_similarity(
            found_image_path,
            lost_image_path
        )


        # AI class comparison
        if prediction.lower() == found_prediction.lower():

            class_score = 100

        else:

            class_score = 0


        # Final score
        # Image = 70%
        # AI class = 30%

        final_score = (
            image_score * 0.70
        ) + (
            class_score * 0.30
        )


        if final_score > best_score:

            best_score = final_score

            best_match = {
                "id": item_id,
                "item_name": item_name,
                "description": description,
                "lost_date": lost_date,
                "location": location,
                "contact": contact,
                "image_name": image_name,
                "prediction": prediction,
                "confidence": confidence,
                "image_score": image_score,
                "match_score": final_score
            }


    return best_match


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("home.html")


@app.route("/lost")
def lost():

    return render_template("index.html")

# =========================================================
# LOST ITEM UPLOAD
# =========================================================

@app.route("/upload", methods=["POST"])
def upload():

    item_name = request.form.get(
        "item_name"
    )

    description = request.form.get(
        "description"
    )

    lost_date = request.form.get(
        "lost_date"
    )

    location = request.form.get(
        "location"
    )

    contact = request.form.get(
        "contact"
    )

    uploaded_file = request.files.get(
        "image"
    )


    if not uploaded_file:

        return "Please upload an image."


    if uploaded_file.filename == "":

        return "Please select an image."


    # Save image
    image_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.filename
    )

    uploaded_file.save(
        image_path
    )


    # AI prediction
    prediction, confidence = predict_image(
        image_path
    )


    # Save to database
    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO items
        (
            item_name,
            description,
            lost_date,
            location,
            contact,
            image_name,
            prediction,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        item_name,
        description,
        lost_date,
        location,
        contact,
        uploaded_file.filename,
        prediction,
        confidence
    ))

    conn.commit()

    conn.close()


    # Result page
    return render_template(
        "result.html",

        item_name=item_name,

        description=description,

        lost_date=lost_date,

        location=location,

        contact=contact,

        image_name=uploaded_file.filename,

        prediction=prediction,

        confidence=round(
            confidence,
            2
        )
    )
# =========================================================
# VIEW ALL UPLOADED ITEMS
# =========================================================

@app.route("/my_uploads")
def my_uploads():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # LOST ITEMS
    cursor.execute("""
        SELECT
            id,
            item_name,
            description,
            lost_date,
            location,
            contact,
            image_name,
            prediction,
            confidence
        FROM items
        ORDER BY id DESC
    """)

    lost_items = cursor.fetchall()


    # FOUND ITEMS
    cursor.execute("""
        SELECT
            id,
            item_name,
            description,
            found_date,
            location,
            contact,
            image_name,
            prediction,
            confidence
        FROM found_items
        ORDER BY id DESC
    """)

    found_items = cursor.fetchall()

    conn.close()


    return render_template(
        "my_uploads.html",
        lost_items=lost_items,
        found_items=found_items
    )

# =========================================================
# FOUND PAGE
# =========================================================

@app.route("/found")
def found():

    return render_template(
        "found.html"
    )


# =========================================================
# FOUND ITEM UPLOAD + MATCHING
# =========================================================

@app.route(
    "/found_upload",
    methods=["POST"]
)
def found_upload():

    item_name = request.form.get(
        "item_name"
    )

    description = request.form.get(
        "description"
    )

    found_date = request.form.get(
        "found_date"
    )

    location = request.form.get(
        "location"
    )

    contact = request.form.get(
        "contact"
    )

    uploaded_file = request.files.get(
        "image"
    )


    if not uploaded_file:

        return "Please upload an image."


    if uploaded_file.filename == "":

        return "Please select an image."


    # Save found image
    image_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.filename
    )

    uploaded_file.save(
        image_path
    )


    # AI prediction
    prediction, confidence = predict_image(
        image_path
    )


    # Save found item
    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO found_items
        (
            item_name,
            description,
            found_date,
            location,
            contact,
            image_name,
            prediction,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        item_name,
        description,
        found_date,
        location,
        contact,
        uploaded_file.filename,
        prediction,
        confidence
    ))

    conn.commit()

    conn.close()


    # =====================================================
    # FIND BEST MATCH
    # =====================================================

    print("\nSearching for matching lost item...")

    best_match = find_best_match(
        image_path,
        prediction
    )


    # =====================================================
    # NO MATCH
    # =====================================================

    if best_match is None:

        return render_template_string("""
        <!DOCTYPE html>

        <html>

        <head>

            <title>Smart Lost & Found</title>

            <style>

                body {
                    font-family: Arial;
                    background: linear-gradient(
                        135deg,
                        #4facfe,
                        #00f2fe
                    );

                    min-height: 100vh;

                    display: flex;

                    justify-content: center;

                    align-items: center;
                }

                .card {

                    background: white;

                    padding: 40px;

                    border-radius: 20px;

                    width: 500px;

                    text-align: center;

                    box-shadow: 0 10px 30px
                    rgba(0,0,0,0.2);
                }

                h1 {
                    color: #333;
                }

                .prediction {

                    font-size: 24px;

                    color: #1976d2;

                    font-weight: bold;

                    margin: 20px;
                }

                .btn {

                    display: inline-block;

                    margin-top: 20px;

                    padding: 12px 25px;

                    background: #1976d2;
                    background: white;

                    padding: 40px;

                    border-radius: 20px;

                    width: 500px;

                    text-align: center;

                    box-shadow: 0 10px 30px
                    rgba(0,0,0,0.2);
                }

                h1 {
                    color: #333;
                }

                .prediction {

                    font-size: 24px;

                    color: #1976d2;

                    font-weight: bold;

                    margin: 20px;
                }

                .btn {

                    display: inline-block;

                    margin-top: 20px;

                    padding: 12px 25px;

                    background: #1976d2;

                    color: white;

                    text-decoration: none;

                    border-radius: 10px;
                }

            </style>

        </head>

        <body>

            <div class="card">

                <h1>🔍 No Matching Lost Item</h1>

                <p class="prediction">
                    AI Prediction: {{ prediction }}
                </p>

                <p>
                    No lost item is currently available
                    for matching.
                </p>

                <a href="/found" class="btn">
                    Upload Another Found Item
                </a>

            </div>

        </body>

        </html>
        """, prediction=prediction)


    # =====================================================
    # MATCH FOUND
    # =====================================================

    return render_template_string("""

    <!DOCTYPE html>

    <html>

    <head>

        <title>Smart Lost & Found - Match Result</title>

        <style>

            * {
                box-sizing: border-box;
                font-family: Arial, sans-serif;
            }

            body {

                margin: 0;

                min-height: 100vh;

                background: linear-gradient(
                    135deg,
                    #4facfe,
                    #00f2fe
                );

                display: flex;

                justify-content: center;

                align-items: center;

                padding: 30px;
            }

            .card {

                background: white;

                width: 600px;

                max-width: 95%;

                padding: 30px;

                border-radius: 20px;

                box-shadow: 0 10px 35px
                rgba(0,0,0,0.2);

                text-align: center;
            }

            h1 {

                color: #2e7d32;

                margin-bottom: 20px;
            }

            .success {

                font-size: 20px;

                font-weight: bold;

                color: #2e7d32;

                margin-bottom: 20px;
            }

            .score {

                background: #e8f5e9;

                padding: 20px;

                border-radius: 15px;

                margin: 20px 0;
            }

            .score h2 {

                margin: 0;

                color: #2e7d32;

                font-size: 32px;
            }

            .item {

                text-align: left;

                background: #f5f5f5;

                padding: 20px;

                border-radius: 15px;

                margin-top: 20px;
            }

            .item p {

                margin: 10px 0;

                font-size: 16px;
            }

            .item img {

                width: 220px;

                max-width: 100%;

                border-radius: 15px;

                margin-bottom: 15px;
            }

            .prediction {

                color: #1976d2;

                font-weight: bold;

                font-size: 20px;
            }

            .btn {

                display: inline-block;

                margin-top: 20px;

                padding: 12px 25px;

                background: #1976d2;

                color: white;

                text-decoration: none;

                border-radius: 10px;
            }

        </style>

    </head>

    <body>

        <div class="card">

            <h1>🎉 Match Found!</h1>

            <div class="success">
                A possible lost item match was found.
            </div>

            <div class="score">

                <p>Overall Match Score</p>

                <h2>
                    {{ match_score }}%
                </h2>

            </div>


            <div class="item">

                <img
                    src="/static/uploads/{{ image_name }}"
                    alt="Lost Item"
                >

                <p>
                    📦 <strong>Item Name:</strong>
                    {{ item_name }}
                </p>

                <p>
                    📝 <strong>Description:</strong>
                    {{ description }}
                </p>

                <p>
                    📅 <strong>Lost Date:</strong>
                    {{ lost_date }}
                </p>

                <p>
                    📍 <strong>Location:</strong>
                    {{ location }}
                </p>

                <p>
                    📞 <strong>Contact:</strong>
                    {{ contact }}
                </p>

                <p class="prediction">
                    🔎 AI Prediction:
                    {{ prediction }}
                </p>

                <p>
                    📊 AI Confidence:
                    {{ confidence }}
                  </p>

                <p>
                    🖼️ <strong>Image Similarity:</strong>
                    {{ image_score }}%
                </p>

            </div>

            <a href="/" class="btn">
                Upload Lost Item
            </a>

            <a href="/found" class="btn">
                Upload Another Found Item
            </a>

        </div>

    </body>

    </html>

    """,

    item_name=best_match["item_name"],

    description=best_match["description"],

    lost_date=best_match["lost_date"],

    location=best_match["location"],

    contact=best_match["contact"],

    image_name=best_match["image_name"],

    prediction=best_match["prediction"],

    confidence=round(
        best_match["confidence"],
        2
    ),

    image_score=round(
        best_match["image_score"],
        2
    ),

    match_score=round(
        best_match["match_score"],
        2
    ))


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )