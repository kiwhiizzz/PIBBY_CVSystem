import json
from datetime import datetime

def package_output (bbox, emotion_label, emotion_confidence, distance_level, attentive):
    time = datetime.now().isoformat()
    face_detected = bbox is not None

    if not face_detected:
        output = {
            "time" : time,
            "face_detected" : False, 
            "emotion" : None,
            "attetion" : None,
            "bounding_box" : None
        }

    else :
        output = {
            "time" : time,
            "face_detected" : True, 
            "emotion" : {
                "label" : emotion_label,
                "confidence" : emotion_confidence
            },
            "attetion" : {
                "distance_level" : distance_level,
                "attentive" : attentive
            },
            "bounding_box" : bbox
        }

    return output

def json_string(output_dict):
    return json.dumps(output_dict, indent=2)

