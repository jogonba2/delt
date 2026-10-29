import math

INFO_FILENAME = "model.json"
EMBEDDINGS_FILENAME = "label_embeddings.safetensors"

DEFAULT_LOGIT_SCALE = math.log(1 / 0.07)
MAX_LOGIT_SCALE = math.log(100)
