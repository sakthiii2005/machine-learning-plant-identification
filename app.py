from flask import Flask, request, redirect, session, url_for, render_template, jsonify
import os
import json
import mysql.connector
from werkzeug.utils import secure_filename
from urllib.parse import parse_qs, urlencode, urlparse
from PIL import Image
import pickle
import tensorflow as tf
import numpy as np

import os
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join("D:", os.path.sep, "Web Development", "Leaf"),
    shuffle=True,
    batch_size=32,
    image_size=(224, 224),
)
labels = np.array(dataset.class_names).reshape(-1,1)

app = Flask(__name__)

app.secret_key = 'sakthi1234'

uploads_dir = os.path.join(os.getcwd(), 'uploads')

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    passwd='Magic@2005',
    database='plant'
)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(uploads_dir, filename))

            with open('Cnn.pickle', 'rb') as saved_model:
                loaded_model = pickle.load(saved_model)

            
            image_path = os.path.join(uploads_dir, filename)
            imgs = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224)) 
            imgs = tf.keras.preprocessing.image.img_to_array(imgs)
            imgs = tf.expand_dims(imgs, 0)
    
            predict = loaded_model.predict(imgs)
            score = tf.nn.sigmoid(predict[0])

            prediction_result_index = np.argmax(score)

            prediction_result = labels[prediction_result_index]
            cursor = conn.cursor()
            cursor.execute("SELECT plant_name,plant_type,plant_use,plant_location,plant_link FROM details WHERE plant_name = %s", (str(prediction_result[0]),))
            result = cursor.fetchall()
            
            cursor.close()
            
            data_to_send = [{"Name": row[0], "Type": row[1], "Use": row[2], "Location": row[3],"Link":row[4]} for row in result]
            
            session['prediction_data'] = data_to_send
            return redirect(url_for('display_results'))
    return render_template('index.html')

@app.route('/results/')
def display_results():
    prediction_data = session.get('prediction_data')
    session.pop('prediction_data', None)
    
    return render_template('results.html', data=prediction_data)

if __name__ == '__main__':
    app.run(debug=True)
