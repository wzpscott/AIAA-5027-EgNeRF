import os

for scene in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic"]:
    os.system(f"ns-train egnerf --method-name EventNeRF --experiment-name {scene} --data {scene} --pipeline.model.use-original-event-nerf True")
    os.system(f"ns-train egnerf --method-name EgNeRF --experiment-name {scene} --data {scene}")
    os.system(f"ns-train egnerf --method-name NeRF --experiment-name {scene} --data {scene} --pipeline.model.event-loss-mult 0")