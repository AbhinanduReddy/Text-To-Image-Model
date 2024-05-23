import torch
from sklearn.metrics import recall_score

# Assume `model`, `valid_loader`, `image_embeddings`, and `text_embeddings` are already defined

def evaluate_image_text_retrieval(model, valid_loader, top_k=1):
    model.eval()
    all_image_embeddings = []
    all_text_embeddings = []
    all_captions = []

    with torch.no_grad():
        for batch in valid_loader:
            images = batch['image'].to(CFG.device)
            captions = batch['caption']
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)

            image_features = model.image_encoder(images)
            text_features = model.text_encoder(input_ids, attention_mask)

            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)

            all_image_embeddings.append(image_embeddings)
            all_text_embeddings.append(text_embeddings)
            all_captions.extend(captions)

    all_image_embeddings = torch.cat(all_image_embeddings)
    all_text_embeddings = torch.cat(all_text_embeddings)

    retrieval_scores = []
    for i, text_embedding in enumerate(all_text_embeddings):
        similarities = torch.matmul(text_embedding, all_image_embeddings.T)
        top_k_indices = torch.topk(similarities, k=top_k).indices
        correct = i in top_k_indices
        retrieval_scores.append(correct)

    r_at_k = sum(retrieval_scores) / len(retrieval_scores)
    return r_at_k

r_at_1 = evaluate_image_text_retrieval(model, valid_loader, top_k=1)
print(f'R@1: {r_at_1:.4f}')
