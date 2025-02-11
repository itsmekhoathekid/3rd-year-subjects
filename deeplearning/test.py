import json 

with open(r'C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\content_7.json','r',encoding='utf-8') as f:
    original_data_dict = json.load(f)


import json
import numpy as np
import hashlib
# from face_detection_model import detect_faces
# from face_embedding_model import extract_face_embeddings
# from image_loading_module import load_image



new_data_dict = {}
for article_id, article_data in original_data_dict.items():
    context = article_data["context"]
    for image_data in article_data["images"]:
        image_path = image_data["path"]
        caption = image_data["caption"]
        # Tạo hash_id cho hình ảnh
        hash_id = hashlib.md5(image_path.encode('utf-8')).hexdigest()
        new_data_dict[hash_id] = {
            "caption": caption,
            'image_path': image_path,
            "context": context,
            "face_emb_dir": [],
            "obj_emb_dir": [],
            # Các trường thông tin khác nếu cần
        }

        # # Xử lý hình ảnh
        # image = load_image(image_path)
        # faces = detect_faces(image)
        # if faces:
        #     embeddings = extract_face_embeddings(faces)
        #     # Lưu embeddings vào tệp numpy
        #     face_emb_path = f"faces/{hash_id}.npy"
        #     np.save(face_emb_path, embeddings)
        #     # Cập nhật face_emb_dir
        #     new_data_dict[hash_id]["face_emb_dir"] = [face_emb_path]
        # else:
        #     new_data_dict[hash_id]["face_emb_dir"] = []

        # Tương tự cho obj_emb_dir nếu cần

# Lưu dữ liệu mới
with open(r'C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\restructured_content_7.json', 'w', encoding='utf-8') as f:
    json.dump(new_data_dict, f, ensure_ascii=False, indent=2)
