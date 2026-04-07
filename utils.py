import torch

def clean_text(text):
    """脏数据终极清洗"""
    if text is None:
        return "empty_content"
    text = str(text).strip()
    text = " ".join(text.split())
    if len(text) < 10:
        return "short_content_" + text
    return text

def mean_pooling(token_embeddings, attention_mask):
    """Token向量池化"""
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

@torch.no_grad()
def get_embeddings(texts, tokenizer, model, device, chunk_size):
    """批量获取文本向量"""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=chunk_size, return_tensors="pt").to(device)
    outputs = model(**inputs)
    emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

@torch.no_grad()
def rewrite_query(query, tokenizer, model, device, mock=False):
    """查询重写"""
    if mock:
        return f"[MOCK] {query}"
    inputs = tokenizer(f"Optimize query: {query}", return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)