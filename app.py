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


#2
# تعيين مفتاح واجهة برمجة التطبيقات لـ groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# إعداد نموذج المحادثة
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

try:
    # تحميل نموذج YOLO لتصنيف الزحام
    classification_model = load_model('classification_model.h5')
    print("classification model loaded successfully!")
    
    # تحميل نموذج YOLOv8 لاكتشاف السيارات
    model_yolo = YOLO('yolov8n.pt')  # استبدل 'yolov8n.pt' بمسار النموذج الخاص بك
    print("YOLOv8 model for car detection loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# إعداد القالب الخاص بالمحادثات
prompt_template = PromptTemplate(
    input_variables=["question", "video_analysis", "car_count", "highest_stop_time"],
    template="""
أنت مساعد ذكاء اصطناعي متخصص في تحليل الفيديوهات المرورية. السؤال المطروح هو: {question}

إليك نتائج تحليل الفيديو المتاحة:
- تفاصيل تحليل الفيديو: {video_analysis}

إذا كان السؤال يتعلق بعدد السيارات المكتشفة، فالإجابة هي: {car_count}.
إذا كان السؤال متعلقًا بزمن توقف السيارات، فالإجابة هي: {highest_stop_time}.

في حال كان السؤال غير مرتبط بأي من عدد السيارات أو زمن التوقف، يرجى تقديم الإجابة بناءً على البيانات المتوفرة من تحليل الفيديو.

ملاحظة: إذا كان الاستفسار خارج نطاق الازدحام المروري أو تحليل الفيديوهات المرورية، وضح للمستخدم أنك مختص فقط في هذا المجال، وأنك هنا لتقديم معلومات متعلقة بالازدحام المروري فقط.
اختصر اجابتك ولا تكررها
الرجاء الإجابة باللغة العربية.
"""
)

MODEL_combined = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

app = Flask(__name__)

# إعداد مجلد حفظ الفيديوهات
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')
#############################       save_uploaded_file     #################################
def save_uploaded_file(file_data, filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(video_path, 'wb') as f:
        f.write(file_data)
    print(f"Video saved to {video_path}")
    return video_path
#############################       analyze_video     #################################
def preprocess_frame_for_classification(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # تطبيع البيانات
    return img_array

def analyze_video(video_path):
    video = cv2.VideoCapture(video_path)
    congestion_predictions = []  # قائمة لتجميع تنبؤات الازدحام

    success, frame = video.read()
    while success:

            # تصنيف الازدحام في الإطار الحالي باستخدام نموذج YOLO
        img_array = preprocess_frame_for_classification(frame)

    # التنبؤ بنوع الازدحام باستخدام نموذج التصنيف
        prediction = classification_model.predict(img_array)
        class_label = np.argmax(prediction)

    # الحصول على نوع الازدحام بناءً على التصنيف
        congestion_type = congestion_label(class_label)
        congestion_predictions.append(congestion_type)

        success, frame = video.read()

    video.release()

    # حساب التصنيف الأكثر شيوعًا للازدحام باستخدام max
    most_common_congestion = None
    if congestion_predictions:
        most_common_congestion = max(set(congestion_predictions), key=congestion_predictions.count)
    else:
        most_common_congestion = "Unknown"
    
    return f"Congestion type: {most_common_congestion}", most_common_congestion

#############################       congestion_label     #################################
def congestion_label(class_label):
    congestion_map = {
        0: "زحام مفتعل بسبب اشخاص",
        1: "زحام طبيعي",
        2: "لا يوجد زحام"
    }
    return congestion_map.get(class_label, "Unknown")
#############################       track_unique_cars     #################################
def track_unique_cars(video_path):

    # إعداد Deep SORT مع تحسين المعلمات
    tracker = DeepSort(max_age=15, n_init=3, nn_budget=70)

    # فتح الفيديو
    video = cv2.VideoCapture(video_path)

    # إعداد مجموعة لتخزين معرفات المركبات الفريدة
    unique_car_ids = set()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # كشف السيارات باستخدام YOLOv8 unique_car_ids = set()
        results = model_yolo(frame)

        # تصفية السيارات فقط
        car_detections = []
        for result in results[0].boxes:
            if result.cls == 2:  # التصنيف 2 يشير إلى "car" في مجموعات YOLO
                confidence = float(result.conf)
                if confidence > 0.5:  # تصفية الكائنات بناءً على الثقة
                    x1, y1, x2, y2 = map(int, result.xyxy[0])  # إحداثيات صندوق الكشف
                    width, height = x2 - x1, y2 - y1
                    car_detections.append([[x1, y1, width, height], confidence])

        # تتبع السيارات باستخدام Deep SORT
        tracks = tracker.update_tracks(car_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id  # معرف السيارة
            bbox = track.to_ltrb()  # الحصول على إحداثيات الصندوق (يسار، أعلى، يمين، أسفل)

            # إضافة المعرف الفريد لكل سيارة إلى مجموعة unique_car_ids
            unique_car_ids.add(track_id)

            # رسم الصندوق حول السيارة
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # تحرير موارد الفيديو
    video.release()
    cv2.destroyAllWindows()

    # طباعة عدد السيارات الفريدة
    print(f"عدد السيارات الفريدة: {len(unique_car_ids)}")

    return len(unique_car_ids)

#############################       calculate_time     #################################
def calculate_time(video_path):
    cap = cv2.VideoCapture(video_path)
    highest_time = 0

    # إعداد الفيديو لحفظ الإخراج
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # إعداد تعقب المركبات
    vehicle_positions = defaultdict(list)  # لحفظ مراكز المركبات
    vehicle_stop_time = defaultdict(int)  # لحفظ زمن التوقف لكل مركبة
    MOVEMENT_THRESHOLD = 10  # عتبة الحركة (بالبكسل) لتحديد ما إذا كانت المركبة تتحرك

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # اكتشاف المركبات باستخدام YOLOv8
        results = model_yolo(frame)

        # تعقب المركبات واكتشاف توقفها
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy.cpu().numpy()[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            vehicle_id = f'{int(x1)}_{int(y1)}'  # تحديد هوية فريدة لكل مركبة باستخدام إحداثياتها

            # إذا كانت المركبة قد تم تعقبها من قبل
            if vehicle_positions[vehicle_id]:
                prev_center_x, prev_center_y = vehicle_positions[vehicle_id][-1]
                # التحقق مما إذا كانت المركبة قد تحركت (إذا كان الفرق في المركز أكبر من MOVEMENT_THRESHOLD)
                if (abs(center_x - prev_center_x) < MOVEMENT_THRESHOLD) and (abs(center_y - prev_center_y) < MOVEMENT_THRESHOLD):
                    vehicle_stop_time[vehicle_id] += 1  # زيادة زمن التوقف بالإطارات
                else:
                    vehicle_stop_time[vehicle_id] = 0  # إعادة تعيين زمن التوقف إذا تحركت المركبة
            else:
                vehicle_stop_time[vehicle_id] = 0  # إذا كانت المركبة جديدة

            # حساب زمن التوقف بالثواني لكل مركبة
            stop_seconds = vehicle_stop_time[vehicle_id] / fps  # تحويل زمن التوقف إلى ثواني
            if highest_time < stop_seconds:
                highest_time = stop_seconds

            # رسم مستطيل حول المركبة وإظهار وقت التوقف إذا توقفت
            color = (0, 255, 0)
            if stop_seconds > MOVEMENT_THRESHOLD / fps:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'Stopped: {stop_seconds:.2f}s', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # تحديث مركز المركبة
            vehicle_positions[vehicle_id].append((center_x, center_y))


    # تحرير الفيديو وإغلاق النوافذ
    cap.release()
    
    # إرجاع أعلى زمن توقف بالثواني مباشرة
    return highest_time
#############################       upload_video     #################################
@app.route('/upload', methods=['POST'])
def upload_video():

    global video_analysis_result, total_car_count, most_common_congestion, highest_stop_time
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected video'}), 400

    try:
        video_data = video.read()
        video_filename = video.filename
        video_path = save_uploaded_file(video_data, video_filename)
    except Exception as e:
        return jsonify({'error': f'Error saving video: {str(e)}'}), 500

    try:
        # تحليل الفيديو
        video_analysis_result, most_common_congestion = analyze_video(video_path)
        
        # حساب زمن التوقف باستخدام دالة calculate_time
        highest_stop_time = calculate_time(video_path)
        total_car_count = track_unique_cars(video_path)
        
        if video_analysis_result is not None:
            return jsonify({
                'message': 'Video analyzed successfully.',
                'result': video_analysis_result,
                'car_count': total_car_count,
                'congestion_type': most_common_congestion,
                'highest_stop_time_seconds': highest_stop_time,  # زمن التوقف بالثواني
                'video_url': f'/static/{video_filename}',
            })
        else:
            return jsonify({'error': 'Error analyzing video: No valid frames analyzed.'}), 500
    except Exception as e:
        return jsonify({'error': f'Error during video analysis: {str(e)}'}), 500
    

@app.route('/ask', methods=['POST'])
#############################       ask_user_question     #################################
def ask_user_question():
    
    if not video_analysis_result:
        return jsonify({'error': 'No video has been analyzed yet.'}), 400

    user_question = request.json.get('question')

    # تحقق مما إذا كان السؤال يتعلق بعدد السيارات أو زمن التوقف
    car_question_keywords = ["عدد السيارات", "كم سيارة", "السيارات المكتشفة"]
    stop_time_keywords = ["زمن التوقف", "مدة التوقف", "توقف المركبات"]

    if any(keyword in user_question for keyword in car_question_keywords):
        response = f"عدد السيارات المكتشفة هو {total_car_count}."
    elif any(keyword in user_question for keyword in stop_time_keywords):
        response = f"أطول مدة توقف هي {highest_stop_time:.2f} ثانية."
    else:
        # استدعاء النموذج للإجابة على الأسئلة الأخرى المتعلقة بالازدحام
        response = MODEL_combined.run({
            "question": user_question,
            "video_analysis": str(video_analysis_result),
            "congestion_type": most_common_congestion,
            "car_count": total_car_count,
            "highest_stop_time": highest_stop_time
        })

    return jsonify({'answer': response})
#############################       serve_video     #################################
@app.route('/static/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8000, debug=False)
    except SystemExit:
        pass