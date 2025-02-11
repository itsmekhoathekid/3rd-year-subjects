import json
import os
from py_vncorenlp import VnCoreNLP
from collections import defaultdict
import re
import numpy as np
# from extract_face import extract_faces, get_face_embedding, save_face_embeddings
# from extract_object import detect_objects, extract_object_embedding

def preprocess(text):
    cleaned_text = re.sub(r'\(Ảnh.*?\)', '', text)
    return cleaned_text

model = VnCoreNLP(save_dir=r'D:\Code\Python\Image_captioning\VACNIC\VnCoreNLP', annotators=["wseg","ner","pos"])

def extract_entities(text, model):
    """
    Sử dụng VnCoreNLP để trích xuất các thực thể từ văn bản.
    Trả về một dictionary với các loại thực thể.
    """
    entities = defaultdict(set)  # Sử dụng set để loại bỏ trùng lặp
    if not text.strip():
        return entities  # Trả về empty nếu text rỗng

    try:
        annotated_text = model.annotate_text(text)
    except Exception as e:
        print(f"Lỗi khi annotating text: {e}")
        return entities

    # Danh sách nhãn ánh xạ chuẩn
    label_mapping = {
        "PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION",
        "GPE": "GPE", "NORP": "NORP", "MISC": "MISC",
        "B-PER": "PERSON", "I-PER": "PERSON",
        "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
        "B-LOC": "LOCATION", "I-LOC": "LOCATION",
        "B-GPE": "GPE", "I-GPE": "GPE",
        "B-NORP": "NORP", "I-NORP": "NORP",
        "B-MISC": "MISC", "I-MISC": "MISC"
    }

    for sent in annotated_text:
        for word in annotated_text[sent]:
            ent_type = label_mapping.get(word.get('nerLabel', ''), '')  
            ent_text = word.get('wordForm', '').strip()
            if ent_type and ent_text:
                entities[ent_type].add(ent_text) 

    return {key: list(value) for key, value in entities.items()}

def find_name_positions(caption, names):
    """
    Tìm vị trí của các tên trong caption.
    Trả về danh sách các cặp [start, end].
    """
    positions = []
    for name in names:
        start = caption.find(name)
        while start != -1:
            end = start + len(name)
            positions.append([start, end])
            start = caption.find(name, end)  
    return positions

def process_dataset(input_json_path, output_json_path, model):
    """
    Đọc dữ liệu từ input_json_path, xử lý và lưu vào output_json_path.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {}

    for hash_id, content in data.items():
        new_entry = {}

        # Lấy các hình ảnh và captions
        image_path = content.get("image_path", [])


        # trích xuất face
        # faces = extract_faces(image_path)
        # faces_embbed = []
        # if faces:
        #     # Embedding hết face trong faces
        #     for face in faces:
        #         face_emb = get_face_embedding(face)
        #         faces_embbed.append(face_emb)

        #     face_emb_path = os.path.join("D:/Code/Python/Image_captioning/VACNIC/src/data/faces", f"{hash_id}.npy")
        #     np.save(face_emb_path, faces_embbed)
        #     new_entry["face_emb_dir"] = face_emb_path
        # else: 
        #     new_entry["face_emb_dir"] = []
        new_entry["face_emb_dir"] = []
        # objects, _ = detect_objects(image_path)
        # objects_embbed = []
        # if objects:
        #     # Embedding hết object trong objects
        #     for object in objects:

        #         object_emb = extract_object_embedding(image_path,object)
        #         objects_embbed.append(object_emb)

        #     object_emb_path = os.path.join("D:/Code/Python/Image_captioning/VACNIC/src/data/objects", f"{hash_id}.npy")
        #     np.save(object_emb_path, objects_embbed)
        #     new_entry["obj_emb_dir"] = object_emb_path
        # else: 
        #     new_entry["obj_emb_dir"] = []
        new_entry["obj_emb_dir"] = []
        caption = content.get("caption", [])
        

        # Lấy các câu trong context
        contexts = content.get("context", [])
        context_text = " ".join(contexts)

        # Trích xuất thực thể từ context
        context_entities = extract_entities(context_text, model)
        names_art = context_entities.get("PERSON", [])
        org_norp_art = context_entities.get("ORGANIZATION", []) + context_entities.get("NORP", [])
        gpe_loc_art = context_entities.get("GPE", []) + context_entities.get("LOCATION", [])

        # Trích xuất thực thể từ captions
        captions_text = caption
        caption_entities = extract_entities(captions_text, model)
        names_caption = caption_entities.get("PERSON", [])
        org_norp_caption = caption_entities.get("ORGANIZATION", []) + caption_entities.get("NORP", [])
        gpe_loc_caption = caption_entities.get("GPE", []) + caption_entities.get("LOCATION", [])

        # trích xuất thực thể từ captions
        ner_cap = []
        for ent_type, entities in caption_entities.items():
            ner_cap.extend(entities)
        new_entry["ner_cap"] = list(set(ner_cap))  # Thêm trường ner_cap

        # kết hợp context và caption để tạo named_entities
        named_entites = set()
        for ent_type, entities in context_entities.items():
            named_entites.update(entities)
        for ent_type, entities in caption_entities.items():
            named_entites.update(entities)
        new_entry["named_entites"] = list(named_entites)

        # Tạo các trường mới với loại thực thể duy nhất
        new_entry["names_art"] = list(set(names_art))
        new_entry["org_norp_art"] = list(set(org_norp_art))
        new_entry["gpe_loc_art"] = list(set(gpe_loc_art))

        # Kết hợp thực thể từ context và captions
        new_entry["names"] = list(set(names_art + names_caption))
        new_entry["org_norp"] = list(set(org_norp_art + org_norp_caption))
        new_entry["gpe_loc"] = list(set(gpe_loc_art + gpe_loc_caption))


        # Thêm các trường hiện tại
        new_entry["image_path"] = image_path
        new_entry["caption"] = caption
        new_entry["context"] = contexts


        name_pos_cap = find_name_positions(caption, names_caption)
        new_entry["name_pos_cap"] = name_pos_cap 
        # Các trường khác có thể thêm vào sau nếu cần
        new_entry["sents_byclip"] = []
        new_data[hash_id] = new_entry

    # Lưu dữ liệu đã xử lý vào file mới
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được xử lý và lưu vào {output_json_path}")

if __name__ == "__main__":
    input_json = r"D:\Code\Python\Image_captioning\data\train\restructured_content_7.json" 
    output_json = r"D:\Code\Python\Image_captioning\data\train\restructured_content_7.json"  

    process_dataset(input_json, output_json, model)
    