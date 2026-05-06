import os
import cv2
import numpy as np
import onnxruntime as ort

from config.settings import (
    ANTI_SPOOF_MODEL_PATH,
    ANTI_SPOOF_REAL_THRESHOLD,
    ANTI_SPOOF_FAKE_THRESHOLD,
)


class AntiSpoofModel:
    def __init__(self):
        self.model_available = False
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_height = 128
        self.input_width = 128

        if not os.path.exists(ANTI_SPOOF_MODEL_PATH):
            print(f"Anti-spoof model not found at: {ANTI_SPOOF_MODEL_PATH}")
            return

        try:
            self.session = ort.InferenceSession(
                ANTI_SPOOF_MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )

            input_cfg = self.session.get_inputs()[0]
            output_cfg = self.session.get_outputs()[0]

            self.input_name = input_cfg.name
            self.output_name = output_cfg.name

            input_shape = input_cfg.shape

            if len(input_shape) == 4:
                self.input_height = int(input_shape[2])
                self.input_width = int(input_shape[3])

            self.model_available = True

            print("Anti-spoof model loaded successfully.")
            print("Input:", self.input_name)
            print("Output:", self.output_name)
            print("Input size:", self.input_width, "x", self.input_height)

        except Exception as e:
            print("Invalid anti-spoof model.")
            print(e)

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def softmax(self, x):
        x = x.astype(np.float32)
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def predict(self, frame):
        if not self.model_available:
            return {
                "available": False,
                "score": 0.0,
                "real_score": 0.0,
                "fake_score": 0.0,
                "is_real": False,
                "decision": "ERROR",
            }

        try:
            input_tensor = self.preprocess(frame)

            output = self.session.run(
                [self.output_name],
                {
                    self.input_name: input_tensor,
                },
            )[0]

            probabilities = self.softmax(output)

            # Based on your logs:
            # class 0 appears to be REAL
            # class 1 appears to be FAKE
            real_score = float(probabilities[0][0])
            fake_score = float(probabilities[0][1])

            if real_score >= ANTI_SPOOF_REAL_THRESHOLD and real_score > fake_score:
                decision = "REAL"
                is_real = True

            elif fake_score >= ANTI_SPOOF_FAKE_THRESHOLD and fake_score > real_score:
                decision = "SPOOF"
                is_real = False

            else:
                decision = "UNCERTAIN"
                is_real = False

            print("Anti-spoof real_score:", real_score)
            print("Anti-spoof fake_score:", fake_score)
            print("Anti-spoof decision:", decision)

            return {
                "available": True,
                "score": real_score,
                "real_score": real_score,
                "fake_score": fake_score,
                "is_real": is_real,
                "decision": decision,
            }

        except Exception as e:
            print("Anti-spoof prediction failed:", e)

            return {
                "available": False,
                "score": 0.0,
                "real_score": 0.0,
                "fake_score": 0.0,
                "is_real": False,
                "decision": "ERROR",
            }