from ultralytics import YOLO

model = YOLO("/home/sibwa19/documents/main_course/third/project/ML/ml_project/data/models/best.pt")

results = model.predict(
    source="/home/sibwa19/documents/main_course/third/project/ML/ml_project/data/raw"
    ,project="/home/sibwa19/documents/main_course/third/project/ML/ml_project/data"
    ,name="predict_"
    ,save=True
    ,save_txt=True
    ,device=None
    ,line_thickness=0
    ,boxes=False
)