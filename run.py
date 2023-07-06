import os
import sys
import time
import tqdm
import logging
import pickle
import argparse

sys.path.append("/DATA_17/kjw/DeepFace")
# from deepface import DeepFace
from CustomDeepFace import DeepFace

# from dataset import CustomDatset
import tensorflow.keras.backend as K


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFace_Attributes")
    parser.add_argument(
        "--img_path", default="/DATA_STORAGE/DATASET/Montage_data/train/montage_data.pkl"
    )
    parser.add_argument("--log_output", default="Inference")
    args = parser.parse_args()
    return args


def main(args):
    logger = logging.getLogger(name="Inference.log")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(f"./{args.log_output}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if os.path.splitext(args.img_path)[1].lower() == ".pkl":
        with open(args.img_path, "rb") as f:
            imgs_info = pickle.load(f)
        id2fids, file_info = imgs_info["ids"], imgs_info["file_info"]
    else:
        file_info = [{"f_id": None, "gender": None, "age": None, "img_path": args.img_path}]
    model = DeepFace(actions=["age", "gender", "race"], detector_backend="retinaface")
    g_true = []
    g_pred = []
    a_true = []
    a_pred = []
    j = 0
    logger.info("Infernce Started")
    for i, item in enumerate(tqdm.tqdm(file_info)):
        f_id, gender, age, img_path = item.values()
        # img_path = img_path.replace('DATA_17', 'DATA_STORAGE')
        objs = model.anlayze(img_path)
        # objs = DeepFace.analyze(img_path=args.img_path,
        #                         actions=['age', 'gender', 'race', 'emotion'],
        #                         detector_backend='retinaface'
        #                         )
        # if i % 10 == 0:
        #     K.clear_session()
        if len(objs) == 1:
            obj = objs[0]
        else:
            logger.info(f"idx: {i}, More than two faces??", f_id, img_path)
            print("More than two faces??", i, f_id)
            continue
        if obj["dominant_race"] != "asian":
            logger.warning(f"idx: {i}, is not asian??", f_id, img_path, obj["dominant_race"])
        if "ids2fids" in locals():
            assert f_id == ids2fids[i]
        g_true.append(gender)
        g_pred.append("M" if max(obj["gender"], key=obj["gender"].get) == "Man" else "F")
        a_true.append(age)
        a_pred.append(obj["age"])

        if i % 20 == 0 and i != 0:
            logger.info(f"idx: {j*20}~{i}")
            tmp = {
                "ids": [*range(j * 20, i)],
                "gender_true": g_true[j * 20 : i],
                "gender_pred": g_pred[j * 20 : i],
                "age_true": a_true[j * 20 : i],
                "age_pred": a_pred[j * 20 : i],
            }
            # tmp = {
            #     "ids": [*range(i)],
            #     "gender_true": g_true,
            #     "gender_pred": g_pred,
            #     "age_true": a_true,
            #     "age_pred": a_pred,
            # }
            j += 1
            with open(f"tmp_{i}.pkl", "wb") as f:
                pickle.dump(tmp, f)

    output = {
        "ids": [*range(len(file_info))],
        "gender_true": g_true,
        "gender_pred": g_pred,
        "age_true": a_true,
        "age_pred": a_pred,
    }

    with open("output.pkl", "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
