import torch
from transformers import BertTokenizer, BertForMaskedLM

# 加载预训练的模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# 确保模型处于评估模式
model.eval()

def predict_masked_token(text, tokenizer, model, max_length=512):
    # 对输入文本进行编码
    inputs = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    # 找到 [MASK] token 的位置
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
    # 获取 [MASK] token 位置的预测分数
    mask_token_logits = outputs.logits[0, mask_token_index, :]

    # 找到预测分数最高的 token
    top_token = torch.argmax(mask_token_logits, dim=-1)
    
    # 解码预测的 token
    predicted_token = tokenizer.decode(top_token)
    return predicted_token

# 示例文本
text = "The quick brown fox jumps over the lazy [MASK]."

# 进行预测
predicted_token = predict_masked_token(text, tokenizer, model)
print("Original text:", text)
print("Predicted token for [MASK]:", predicted_token)
