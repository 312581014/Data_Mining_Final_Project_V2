import argparse
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

def main(args):
    behaviors_path = args.behaviors_path
    news_path = args.news_path

    behaviors = pd.read_csv(behaviors_path, sep='\t', names=['user_id', 'clicked_news', 'impressions'], skiprows=1)
    news = pd.read_csv(news_path, sep='\t', usecols=['news_id', 'category', 'subcategory', 'title'], names=['news_id', 'category', 'subcategory', 'title'], skiprows=1)
    
    news['info'] = "news_id is " + news['news_id'] + ": category is " + news['category'] + "; subcategory is " + news['subcategory'] + "; title is " + news['title']
    news_info_dict = news.set_index('news_id')['info'].to_dict()

    behaviors['clicked_news_info'] = behaviors['clicked_news'].apply(lambda ids: [news_info_dict.get(news_id, '') for news_id in ids.split()])

    model_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    data_rows = []
    if args.mode == "train":
        for index, row in behaviors.iterrows():
            input_texts = row["clicked_news_info"]
            batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            mean_embedding = torch.mean(embeddings, dim=0).tolist()
            # print(mean_embedding)

            impressions = row['impressions'].split(' ')
            for impression in impressions:
                parts = impression.split('-')
                if len(parts) == 2:
                    news_id, click_label = parts
                    input_text = news_info_dict.get(news_id, '')
                    batch_dict_for_impression = tokenizer(input_text, max_length=8192, padding=True, truncation=True, return_tensors='pt')
                    outputs_for_impression = model(**batch_dict_for_impression)
                    embeddings_for_impression = outputs_for_impression.last_hidden_state[:, 0].tolist()
                    # print(embeddings_for_impression[0])

                    model_inputs = mean_embedding + embeddings_for_impression[0]
                    # print(f"Length of model_inputs: {len(mean_embedding)} {len(embeddings_for_impression[0])}")
                    row_data = model_inputs + [int(click_label)]
                    data_rows.append(row_data)
                    # print(len(data_rows[0]))
        feature_columns = ['input_' + str(i) for i in range(len(data_rows[0]) - 1)] + ['model_output']
        outputs_df = pd.DataFrame(data_rows, columns=feature_columns)
        outputs_df.to_csv(args.outputs, index=False)
    else:
        for index, row in behaviors.iterrows():
            input_texts = row["clicked_news_info"]
            batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            mean_embedding = torch.mean(embeddings, dim=0).tolist()

            impressions = row['impressions'].split(' ')
            for impression in impressions:
                input_text = news_info_dict.get(impression, '')
                batch_dict_for_impression = tokenizer(input_text, max_length=8192, padding=True, truncation=True, return_tensors='pt')
                outputs_for_impression = model(**batch_dict_for_impression)
                embeddings_for_impression = outputs_for_impression.last_hidden_state[:, 0].tolist()

                model_inputs = mean_embedding + embeddings_for_impression[0]
                # print(f"Length of model_inputs: {len(mean_embedding)} {len(embeddings_for_impression[0])}")
                data_rows.append(model_inputs)
                # print(len(data_rows[0]))
        feature_columns = ['input_' + str(i) for i in range(len(data_rows[0]))]
        outputs_df = pd.DataFrame(data_rows, columns=feature_columns)
        outputs_df.to_csv(args.outputs, index=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--behaviors_path', type=str, required=True)
    parser.add_argument('--news_path', type=str, required=True)
    parser.add_argument('--outputs', type=str, required=True)
    parser.add_argument('--model', type=str, default='Alibaba-NLP/gte-large-en-v1.5')
    args = parser.parse_args()
    main(args)
