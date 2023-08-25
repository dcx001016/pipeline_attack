import torch
import json
import os
import argparse
import tqdm

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m', 
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
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    ## emb
    print('saving embs')
    item = {}
    item['embed_tokens.weight'] = model.state_dict()['model.decoder.embed_tokens.weight']
    item['embed_positions.weight'] = model.state_dict()['model.decoder.embed_positions.weight']
    if 'model.decoder.project_in.weight' in model.state_dict():
        item['project_in.weight'] = model.state_dict()['model.decoder.project_in.weight']
    torch.save(item, os.path.join(save_path, 'pytorch_embs.pt'))

    ## out
    print('saving lm_head')
    item = {}
    item['lm_head.weight'] = model.state_dict()['lm_head.weight']
    if 'model.decoder.project_out.weight' in model.state_dict():
        item['project_out.weight'] = model.state_dict()['model.decoder.project_out.weight']
    # item['final_layer_norm.weight'] = model.state_dict()['model.decoder.final_layer_norm.weight']
    # item['final_layer_norm.bias'] = model.state_dict()['model.decoder.final_layer_norm.bias']
    torch.save(item, os.path.join(save_path, 'pytorch_lm_head.pt'))
    
    print('saving layers')
    for i in tqdm.tqdm(range(0, config.num_hidden_layers)):
        layer_prefix = f'model.decoder.layers.{i}.'

        item = {}

        layer_maps = {k:v for k,v in model.state_dict().items() if k.startswith(layer_prefix)}

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = v

        torch.save(item, os.path.join(save_path, f'pytorch_{i}.pt'))

        del item
    
