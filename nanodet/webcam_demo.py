import cv2
import torch
import time
import numpy as np

from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline


class Predictor:
    def __init__(self, config, model_path):
        load_config(cfg, config)
        self.cfg = cfg

        self.logger = Logger(-1, use_tensorboard=False)

        self.model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location="cpu")
        load_model_weight(self.model, ckpt, self.logger)
        self.model.eval()

        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        h, w = img.shape[:2]

        meta = dict(
            img=img,
            raw_img=img,
            img_info={
                "id": [0],
                "height": [h],
                "width": [w]
            }
        )

        # pipeline
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)

        # 🔥 FIX warp issue
        meta["warp_matrix"] = [np.eye(3, dtype=np.float32)]

        # tensor
        meta["img"] = torch.from_numpy(
            meta["img"].transpose(2, 0, 1)
        ).unsqueeze(0).float()
        start = time.time()
        with torch.no_grad():
            preds = self.model(meta["img"])
            results = self.model.head.post_process(preds, meta)
        
        end = time.time()
        inference_time = end - start

        return results, inference_time


# ---------------- MAIN ----------------

config_path = "config/nanodet-plus-m_320.yml"
model_path = "workspace/nanodet-plus-m_416_checkpoint.ckpt"

predictor = Predictor(config_path, model_path)

cap = cv2.VideoCapture(0)

input_w, input_h = cfg.data.val.input_size
total_time = 0
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #start = time.time()

    results,inference_time = predictor.inference(frame)
    dets = results[0]
    #end = time.time()
    #inference_time = end - start
    fps = 1 / inference_time if inference_time > 0 else 0
    total_time += inference_time
    frame_count += 1
    avg_fps = frame_count / total_time
    if frame_count % 10 == 0:
     print(f"FPS: {fps:.2f} | Avg FPS: {avg_fps:.2f} | Latency: {inference_time*1000:.2f} ms")
    h, w = frame.shape[:2]

    scale_x = w / input_w
    scale_y = h / input_h

    # 🔥 DRAW CORRECTLY
    for label in dets:
        for bbox in dets[label]:
            x1, y1, x2, y2, score = bbox

            if score < 0.4:
                continue

            # scale to original frame
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            name = cfg.class_names[label]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}:{score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    #fps = 1 / (time.time() - start)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("NanoDet Webcam FINAL", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()