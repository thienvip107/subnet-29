import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import base64
import sys
import numpy as np
import random
from loguru import logger
from fastapi import FastAPI, HTTPException
# Import Ray Serve
from ray import serve
from ray.serve.handle import DeploymentHandle
from accelerate import Accelerator

from utils import GenerationRequest

# Define Ray Serve deployment options
NUM_REPLICAS = 2  # Adjust based on your resource availability
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")



class TextToVideoModel:
    def __init__(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b", 
            torch_dtype=torch.float16, 
            variant="fp16"
        )
        self.pipe.to("cuda")
        self.pipe = Accelerator().prepare(self.pipe)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    def preprocess_text(self, text, limit=76):
        tokens = text.split()[:limit]
        return ' '.join(tokens)

    async def __call__(self, request_data: GenerationRequest):
        try:
            seed = request_data.seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            prompt = self.preprocess_text(request_data.text)
            video_frames = self.pipe(prompt, num_inference_steps=25).frames
            video_path = export_to_video(video_frames)
            with open(video_path, 'rb') as video_file:
                video_data = video_file.read()
            video_base64_encoded = base64.b64encode(video_data)
            video_base64_string = video_base64_encoded.decode('utf-8')
            torch.cuda.empty_cache()
            return {'completion': video_base64_string}
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory. Consider reducing the request rate or payload size.")
                raise HTTPException(status_code=503, detail="CUDA out of memory. Try again later.")
            else:
                logger.exception("An error occurred during request processing - Runtime")
                raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception("An error occurred during request processing")
            raise HTTPException(status_code=500, detail=str(e))



class MinerEndpoint:
    def __init__(self, model_handle: DeploymentHandle):
        self.model_handle = model_handle
        self.app = FastAPI()
        self.app.add_api_route("/generate", self.generate, methods=["POST"])

    async def generate(self, request_data: GenerationRequest):
        result = await model_handle.remote(request_data)
        return result



if __name__ == '__main__':
    model_deployment = serve.deployment(
        TextToVideoModel,
        name="deployment",
        num_replicas=2,
        ray_actor_options={"num_gpus": 1},
    )
    serve.run(
        model_deployment.bind(),
        name="deployment-image-challenge",
    )
    model_handle = serve.get_deployment_handle(
        "deployment", "deployment-image-challenge"
    )
    app = MinerEndpoint(model_handle)
    import uvicorn
    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=5005,
    )
