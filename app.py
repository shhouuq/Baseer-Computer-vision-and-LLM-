import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from dotenv import load_dotenv

load_dotenv()


# Setting the API key for Groq
groq_api_key = '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'

# Setting up the conversation model
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    groq_api_key=groq_api_key
)

try:
    # Load classification model 
    classification_model = load_model('/Users/areej/Downloads/classification_model_10.h5')
    print("classification model loaded successfully!")
    
    # Load YOLOv8 model
    model_yolo = YOLO('yolov8n.pt')  
    print("YOLOv8 model for car detection loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# Setting up the prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "video_analysis", "car_count", "highest_stop_time", "most_common_congestion"],
    template="""
أنت مساعد ذكاء اصطناعي متخصص في تحليل الفيديو المروري. السؤال المطروح هو: "{question}"

*تنويه*: إذا كان السؤال لا يتعلق بنوع الزحام، عدد السيارات المكتشفة، أو زمن التوقف، ستكون الإجابة:
"أشكرك على سؤالك، لكنني مختص فقط في تحليل الفيديو المروري. لا أستطيع الإجابة على أسئلة خارج هذا النطاق."

إجابات بناءً على المواضيع المحددة:
- إذا كان السؤال متعلقًا بنوع الزحام: نوع الزحام هو {most_common_congestion}.
- إذا كان السؤال متعلقًا بعدد السيارات: عدد السيارات المكتشفة هو {car_count}.
- إذا كان السؤال متعلقًا بزمن التوقف: أطول زمن توقف هو {highest_stop_time} ثانية.

*يرجى تقديم اجابة احترافية و مختصرة و تجنب التكرار*
*الاجابة باللغة العربية فقط*
"""
)


MODEL_combined = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

app = Flask(_name_)

# Setting up the folder to save videos
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

# Function to save the uploaded video file
def save_uploaded_file(file_data, filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(video_path, 'wb') as f:
        f.write(file_data)
    print(f"Video saved to {video_path}")
    return video_path

def preprocess_frame_for_classification(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

# Function to analyze the video and classify congestion type
def analyze_video(video_path):
    video = cv2.VideoCapture(video_path)
    congestion_predictions = []  # List to store congestion predictions

    success, frame = video.read()
    while success:

        # Classify congestion in the current frame
        img_array = preprocess_frame_for_classification(frame)

        # Predict the congestion type using the classification model
        prediction = classification_model.predict(img_array)
        class_label = np.argmax(prediction)

        # Get the congestion type based on the classification
        congestion_type = congestion_label(class_label)
        congestion_predictions.append(congestion_type)

        success, frame = video.read()

    video.release()

    # Get the most common congestion type using max
    most_common_congestion = None
    if congestion_predictions:
        most_common_congestion = max(set(congestion_predictions), key=congestion_predictions.count)
    else:
        most_common_congestion = "Unknown"
    
    return f"Congestion type: {most_common_congestion}", most_common_congestion

# Map the classification label to the congestion type
CONGESTION_MAP = {
    0: "زحام مفتعل بسبب اشخاص",
    1: "زحام طبيعي",
    2: "لا يوجد زحام"
}
def congestion_label(class_label):
    return CONGESTION_MAP.get(class_label, "Unknown")


# Function to track unique cars using YOLOv8 and Deep SORT
def track_unique_cars(video_path):

    tracker = DeepSort(max_age=3, n_init=20, nn_budget=70)

    video = cv2.VideoCapture(video_path)

    unique_car_ids = set()
    counted_car_ids = set() 

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model_yolo(frame)

        car_detections = []
        for result in results[0].boxes:
            if result.cls == 2:  
                confidence = float(result.conf)
                if confidence > 0.6:  
                    x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())  
                    width, height = x2 - x1, y2 - y1
                    car_detections.append([[x1, y1, width, height], confidence])

        tracks = tracker.update_tracks(car_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id  # Vehicle ID

            # Only count the car if it's not already counted
            if track_id not in counted_car_ids:
                counted_car_ids.add(track_id)
                unique_car_ids.add(track_id)

           
    video.release()
    cv2.destroyAllWindows()

    print(f"Counter : {len(unique_car_ids)}")

    return len(unique_car_ids)


# Function to calculate the stop time of vehicles in the video
def calculate_time(video_path):
    cap = cv2.VideoCapture(video_path)
    highest_time = 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up vehicle tracking
    vehicle_positions = defaultdict(list)  # To store vehicle centers
    vehicle_stop_time = defaultdict(int)  # To store stop time for each vehicle
    MOVEMENT_THRESHOLD = 10  # Pixel threshold to determine if a vehicle is moving

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect vehicles using YOLOv8
        results = model_yolo(frame)

        # Track vehicles and detect if they stop
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy.cpu().numpy()[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            vehicle_id = f'{int(x1)}_{int(y1)}'  

            # If the vehicle has been tracked before
            if vehicle_positions[vehicle_id]:
                prev_center_x, prev_center_y = vehicle_positions[vehicle_id][-1]
                # Check if the vehicle moved (if the difference in center is greater than the threshold)
                if (abs(center_x - prev_center_x) < MOVEMENT_THRESHOLD) and (abs(center_y - prev_center_y) < MOVEMENT_THRESHOLD):
                    vehicle_stop_time[vehicle_id] += 1 
                else:
                    vehicle_stop_time[vehicle_id] = 0 
            else:
                vehicle_stop_time[vehicle_id] = 0  

            # Calculate the stop time in seconds for each vehicle
            stop_seconds = vehicle_stop_time[vehicle_id] / fps 
            if highest_time < stop_seconds:
                highest_time = stop_seconds

            # Draw a rectangle around the vehicle and show stop time if stopped
            color = (0, 255, 0)
            if stop_seconds > MOVEMENT_THRESHOLD / fps:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'Stopped: {stop_seconds:.2f}s', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update the vehicle's position
            vehicle_positions[vehicle_id].append((center_x, center_y))


    # Release the video resources and close windows
    cap.release()
    
    # Return the highest stop time in seconds directly
    return highest_time

@app.route('/upload', methods=['POST'])
# video uploads
def upload_video():

    # Declare global variables to store analysis results
    global video_analysis_result, total_car_count, most_common_congestion, highest_stop_time
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected video'}), 400

    try:
        # Save the uploaded video data
        video_data = video.read()
        video_filename = video.filename
        video_path = save_uploaded_file(video_data, video_filename)
    except Exception as e:
        # If there's an error saving the video, return an error response
        return jsonify({'error': f'Error saving video: {str(e)}'}), 500

    try:
        video_analysis_result, most_common_congestion = analyze_video(video_path)
        
         # Calculate the maximum stop time in the video
        highest_stop_time = calculate_time(video_path)
        total_car_count = track_unique_cars(video_path)

        # If analysis results are available, return the results as a JSON response
        if video_analysis_result is not None:
            return jsonify({
                'message': 'Video analyzed successfully.',
                'result': video_analysis_result,
                'car_count': total_car_count,
                'most_common_congestion': most_common_congestion,
                'highest_stop_time_seconds': highest_stop_time, 
                'video_url': f'/static/{video_filename}',
            })
        
        else:
            return jsonify({'error': 'Error analyzing video: No valid frames analyzed.'}), 500
    except Exception as e:
        return jsonify({'error': f'Error during video analysis: {str(e)}'}), 500
    

@app.route('/ask', methods=['POST'])
# Handle user queries about the analyzed video
def ask_user_question():
    
    # Check if any video has been analyzed yet
    if not video_analysis_result:
        return jsonify({'error': 'No video has been analyzed yet.'}), 400

    # Get the user's question from the request
    user_question = request.json.get('question')

    # Keywords to check
    car_question_keywords = ["عدد السيارات", "كم سيارة", "السيارات المكتشفة"]
    stop_time_keywords = ["زمن التوقف", "مدة التوقف", "توقف المركبات"]

    # If the question is related to car count, provide the car count in the response
    if any(keyword in user_question for keyword in car_question_keywords):
        response = f"عدد السيارات المكتشفة هو {total_car_count}."
    # If the question is related to stop times, provide the highest stop time in the response
    elif any(keyword in user_question for keyword in stop_time_keywords):
        response = f"أطول مدة توقف هي {highest_stop_time:.2f} ثانية."
    # For other traffic-related questions, use the combined model to generate a response
    else:
        response = MODEL_combined.run({
            "question": user_question,
            "video_analysis": str(video_analysis_result),
            "most_common_congestion": most_common_congestion,
            "car_count": total_car_count,
            "highest_stop_time": highest_stop_time
        })
    # Return the answer as a JSON response
    return jsonify({'answer': response})

@app.route('/static/<filename>')
# serve the uploaded video file so it can be viewed
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# Main application 
if _name_ == "_main_":
    try:
        app.run(host="0.0.0.0", port=8000, debug=False)
    except SystemExit:
        pass
