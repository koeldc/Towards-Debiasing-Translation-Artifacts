import torch
from transformers import AutoModel, AutoTokenizer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_data(data, num_egs, output_dir, split):
    model, tokenizer = AutoModel.from_pretrained('bert-base-uncased'), AutoTokenizer.from_pretrained('bert-base-uncased')
    model.eval()
    
    if num_egs == -1:
        data = data
    else:
        data = data[:num_egs]
        
    # Tokenize data
    input_ids = tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # Create iterator
    data_iterator = torch.utils.data.DataLoader(input_ids, batch_size=64, shuffle=False, num_workers=1)
    
    # Iterate over data and encode with BERT
    cls_token_outputs = []
    mean_pooling_outputs = []
    for step, batch in enumerate(data_iterator):
        single_batch = tuple(t.to(device) for t in batch)[0]
        single_batch = single_batch.unsqueeze(0)
        
        # Compute encoding for cls token
        cls_token_output = model(single_batch)[0][:, 0, :]
        cls_token_outputs.append(cls_token_output)
        
        # Compute encoding for mean pooling
        attention_mask = single_batch != tokenizer.pad_token_id
        mean_pooling_output = mean_pooling(model(single_batch), attention_mask)
        mean_pooling_outputs.append(mean_pooling_output)
    
    # Save output
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    np.save(output_dir / f"{split}_cls_token.npy", torch.cat(cls_token_outputs).numpy())
    np.save(output_dir / f"{split}_mean_pooling.npy", torch.cat(mean_pooling_outputs).numpy())

