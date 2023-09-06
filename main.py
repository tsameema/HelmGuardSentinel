from flask import Flask, render_template, request, redirect
import numpy as np
import os, shutil
from ultralytics import YOLO
from pathlib import Path

app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    #function will run the main template
    return render_template('image.html')

'''
    function will upload file, predict it using yolov8 trained model upon custom data,
    and then display its average IOU value, class matching and the plot predicted image 
'''

@app.route('/upload', methods=['POST'])
def upload():
	if request.method == 'POST':
                file = request.files["file"]
                if file.filename == '':  #check if file name not empty
                    return redirect(request.url) 
                if not file.filename.lower().endswith(".jpg"): #check if file is .jpg file
                    return 'Upload File having extension .jpg'
                save_path = os.path.join('tempupload', file.filename)
                file.save(save_path)
                iou_val, match1, path = predict(save_path, file.filename) #predict jpg file
                shutil.copy(os.path.join(path, file.filename), 'static') #move pred image from model folder to html folder
                shutil.rmtree(path) #remove the model folder to control memory issue
                [os.remove(file_path) for file_path in Path('tempupload').iterdir() if os.path.isfile(file_path)] #remove uploaded file from folder
                return render_template('predict.html', output_path=file.filename, iou_val=iou_val, match1=match1)

'''
predict the image using yolov8
'''
def predict(file_path, fname):
    actual = read_ground_truth(fname)
    model =  YOLO('model/best.pt')
    results = model.predict(source = file_path, save = True)
    pred_bbox = results[0].boxes.xywhn.numpy() #normalized x_y center, width and height
    pred_cls  = results[0].boxes.cls.numpy()   #pred class  
    ious, class_match = [],[]
    for j in range(len(actual)):
        gnd = list(map(float, actual[j].strip().split(' ')))  #split the ground truth file
        iou_val = []
        for i in range(len(pred_bbox)):
            iou = calculate_iou(pred_bbox[i], gnd[1:]) #calculate iou between predict and actual bbox
            iou_val.append(iou)

        ious.append(np.max(iou_val)) #append all the iou values that have max value between actual and pred bbox
        idx_max_iou = np.argmax(iou_val) #idx of max iou values
        match1 = True if pred_cls[idx_max_iou].astype(int) == gnd[0] else False  #return True if pred cass match ground truth class
        class_match.append(match1)
    iou_val = f'Average IoU: {np.mean(ious)}' #mean of all iou values
    match1 = 'All classes match' if all(class_match) else 'Classes do not match'
    return iou_val, match1, results[0].save_dir

#read ground truth file
def read_ground_truth(file):
    with open(os.path.join('DATASET/labels/test', file.replace('jpg', 'txt')), 'r') as f:
        actual = f.readlines()
    f.close()
    return actual

#calculate IOU values
def calculate_iou(pred, gnd):
  x1 = max(gnd[0], pred[0])
  y1 = max(gnd[1], pred[1])
  x2 = min(gnd[0] + gnd[2], pred[0] + pred[2])
  y2 = min(gnd[1] + gnd[3], pred[1] + pred[3])
  intersect  = max(0, x2 - x1) * max(0, y2 - y1)

  area_gnd  = gnd[2] * gnd[3]
  area_pred = pred[2] * pred[3]
  union = area_gnd + area_pred - intersect

  return intersect/union
     
		
if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', 5001)))
