from typing import List

import yaml
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(a: List[float], b: List[float]) -> float:
    return float(cosine_similarity([a], [b])[0][0])


class EmbeddingClient:
    def __init__(self, profile: str = "embedding", config_path: str = "configs/profiles.yaml"):
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        profile_cfg = (cfg.get("models") or {}).get(profile, {})
        self.model = profile_cfg.get("model")
        self.client = AsyncOpenAI(
            api_key=profile_cfg.get("api_key"),
            base_url=profile_cfg.get("base_url"),
        )

    async def embed(self, text: str) -> List[float]:
        resp = await self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding
