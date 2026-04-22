# Cell 4: Hugging Face Pipeline（未来OpenVLA直接复用）
from transformers import pipeline
import torch

# 零样本图像分类（VLA视觉理解）
pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
# 下载一张测试图（或用本地）
# 或者直接用文本生成演示
generator = pipeline("text-generation", model="gpt2")  # 轻量
result = generator("机器人看到红苹果，应该", max_length=50)
print(result[0]['generated_text'])