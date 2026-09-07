from src.utils.config import NEAR_LIMIT, FAR_LIMIT
def calculate_face_center(bbox):
    center_x = (bbox["x1"] + bbox["x2"]) / 2
    center_y = (bbox["y1"] + bbox["y2"]) / 2
    return center_x, center_y

def look_at_camera(bbox, frame_width, frame_height, tolerance = 0.2):
    face_center_x,face_center_y =  calculate_face_center(bbox)

    frame_center_x = frame_width / 2
    frame_center_y = frame_height /2

    max_offset_x = frame_width * tolerance
    max_offset_y = frame_height * tolerance 

    offset_x = abs(face_center_x - frame_center_x)
    offset_y = abs(face_center_y - frame_center_y)

    return offset_x <= max_offset_x and offset_y <= max_offset_y

def calculate_distance(distance):

    if distance < NEAR_LIMIT:
        return "near"
    
    elif distance > FAR_LIMIT:
        return "far"
    
    else:
        return "middle"

def estimate_attention(bbox, frame_width, frame_height, distance):
    attentive = look_at_camera(bbox, frame_width, frame_height)
    distance_level = calculate_distance(distance)


    return {
        "attentive" : attentive,
        "distance" : distance_level
    }