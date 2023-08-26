import os
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='bigscience/bloom-7b1', 
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
    item = {}
    item['word_embeddings.weight'] = model.state_dict()['transformer.word_embeddings.weight']
    item['word_embeddings_layernorm.weight'] = model.state_dict()['transformer.word_embeddings_layernorm.weight']
    item['word_embeddings_layernorm.bias'] = model.state_dict()['transformer.word_embeddings_layernorm.bias']
    torch.save(item, os.path.join(save_path, 'pytorch_embs.pt'))


    ## out
    item = {}
    item['lm_head.weight'] = model.state_dict()['lm_head.weight']
    item['ln_f.weight'] = model.state_dict()['transformer.ln_f.weight']
    item['ln_f.bias'] = model.state_dict()['transformer.ln_f.bias']
    torch.save(item, os.path.join(save_path, 'pytorch_lm_head.pt'))

    ## layers
    print('saving layers')
    for i in range(config.n_layer):
        layer_prefix = f'transformer.h.{i}.'

        item = {}

        layer_maps = {k:v for k,v in model.state_dict().items() if k.startswith(layer_prefix)}

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = v

        torch.save(item, os.path.join(save_path, f'pytorch_{i}.pt'))

        del item
