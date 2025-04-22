# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import subprocess
from cog import BasePredictor, Input, Path
import ImageReward as RM
from PIL import Image

IMAGEREWARD_URL = "https://weights.replicate.delivery/default/thudm/ImageReward.pt"
IMAGEREWARD_PATH = "./weights/ImageReward.pt"


class Predictor(BasePredictor):
    test_inputs = {"prompt": "a painting of an ocean with clouds and birds, day time, low depth field effect", "image": "./assets/images/1.webp"}
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(IMAGEREWARD_PATH):
            subprocess.check_call(["pget", IMAGEREWARD_URL, IMAGEREWARD_PATH])
        st = time.time()
        self.model = RM.load(IMAGEREWARD_PATH)
        print(f"model loaded in {time.time() - st}")

    def predict(
        self,
        prompt: str = Input(description="Prompt to score", defaul="a picture"),
        image: Path = Input(description="Image to score", default="assets/images/1.png"),
    ) -> float:
        """Run a single prediction on the model"""
        st = time.time()
        image = Image.open(image)   
        rewards = self.model.score(prompt, [image])
        print(f"eval run in {time.time() - st}")
        return rewards
