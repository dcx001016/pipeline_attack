import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='google/flan-t5-xl', 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, default='checkpoints', 
                        help='model-name')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    config = AutoConfig.from_pretrained(args.model_name)
    config.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(save_path)
    
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    ## emb
    item = {}
    item['shared.weight'] = model.state_dict()['shared.weight']
    torch.save(item, os.path.join(save_path, 'pytorch_embs.pt'))
    
    
    ## out
    item = {}

    item['final_layer_norm.weight'] = model.state_dict()['encoder.final_layer_norm.weight']

    torch.save(item, os.path.join(save_path, 'pytorch_enc_head.pt'))
    
    
    ## out
    item = {}
    item['lm_head.weight'] = model.state_dict()['lm_head.weight']
    item['final_layer_norm.weight'] = model.state_dict()['decoder.final_layer_norm.weight']

    torch.save(item, os.path.join(save_path, 'pytorch_dec_head.pt'))


    ## layers

    for i in tqdm(range(config.num_layers)):
        layer_prefix = f'encoder.block.{i}.'

        item = {}

        layer_maps = {k:v for k,v in model.state_dict().items() if k.startswith(layer_prefix)}
        layer_maps['layer.0.SelfAttention.relative_attention_bias.weight'] = model.state_dict()['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = v

        torch.save(item, os.path.join(save_path, f'pytorch_enc_{i}.pt'))

        del item
        
    for i in tqdm(range(config.num_layers)):
        layer_prefix = f'decoder.block.{i}.'

        item = {}

        layer_maps = {k:v for k,v in model.state_dict().items() if k.startswith(layer_prefix)}
        layer_maps['layer.0.SelfAttention.relative_attention_bias.weight'] = model.state_dict()['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = v

        torch.save(item, os.path.join(save_path, f'pytorch_dec_{i}.pt'))

        del item
