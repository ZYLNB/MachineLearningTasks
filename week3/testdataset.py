from datasets import load_dataset, load_from_disk
import os

# 加载 BookCorpus 数据集
# bookcorpus_dataset = load_dataset('bookcorpus', trust_remote_code=True)

# bookcorpus_dataset.save_to_disk('./bookcorpus')
dataset = load_from_disk('./bookcorpus')
bookcorpus_dataset = dataset['train']
import os

# 创建保存文本文件的目录
os.makedirs('./bookcorpus_texts', exist_ok=True)
# print(bookcorpus_dataset['text'][0])
# 遍历数据集并将每条记录保存为单独的文本文件
# for i, example in enumerate(bookcorpus_dataset['text']):
#     # 每个记录保存为一个文本文件
#     with open(f'./bookcorpus_texts/book_{i}.txt', 'w', encoding='utf-8') as f:
#         f.write(example['text'])

# 将所有记录保存到一个大的文本文件中
with open('./bookcorpus_texts/bookcorpus_all.txt', 'w') as f:
    for i in range(0,10):
        print(bookcorpus_dataset['text'][i])
        f.write(bookcorpus_dataset['text'][i])

