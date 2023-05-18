from comm.comm_utils import *
import datasets
import wandb
from openpyxl import load_workbook

def get_metric(args):
    metrics = []
    if args.task_name == 'cola':
        metric = datasets.load_metric('./metrics/matthews_correlation')
        metrics.append(metric)
        metric = datasets.load_metric('./metrics/accuracy')
        metrics.append(metric)
    if args.task_name in {'qnli', 'qqp', 'mrpc', 'mnli', 'sst2'}:
        metric = datasets.load_metric('./metrics/accuracy')
        metrics.append(metric)
        metric = datasets.load_metric('./metrics/f1')
        metrics.append(metric)
    if args.task_name in {'wikitext', 'wiki103', 'arxiv21'}:
        metric = datasets.load_metric('./metrics/perplexity_custom')
        metrics.append(metric)
    return metrics


def _lm_pred_func(x, y):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    logits = x[:, :-1, :].contiguous()
    labels = y[:, 1:].contiguous()
    loss = loss_fct(logits.transpose(-1, -2), labels).mean(1).detach().cpu()
    return loss

def writetoxlsx(task_name, model, epoch, 
                forward_attack, forward_attack_rate, 
                backward_attack, backward_attack_rate, 
                optimizer,
                perplexity, loss):

    workbook = load_workbook(filename="experiment_lm noise.xlsx")

    sheet = workbook.active

    row_count = str(sheet.max_row + 1)
    sheet["A" + row_count] = task_name
    sheet["B" + row_count] = model
    sheet["C" + row_count] = epoch
    sheet["D" + row_count] = forward_attack
    sheet["E" + row_count] = forward_attack_rate
    sheet["F" + row_count] = backward_attack
    sheet["G" + row_count] = backward_attack_rate
    sheet["H" + row_count] = optimizer
    sheet["I" + row_count] = perplexity
    sheet["J" + row_count] = loss
    workbook.save("experiment_lm noise.xlsx")

def distributed_test_lm_iter(args, pipeline, device, test_data_loader, epoch):
    pipeline.model.eval()
    if get_pipeline_parallel_rank() == 0:
        for i, data in enumerate(test_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.infer_iter(input_ids, None, None)
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        metrics = get_metric(args)
        for i, data in enumerate(test_data_loader):
            labels = data['text'].to(device)
            pipeline.infer_iter(None, labels, None, 
                                metrics=metrics, pred_func=_lm_pred_func)
            
        result = {
            metric.name: metric.compute() for metric in metrics
        }
        print(result)
        
        if epoch == args.n_epochs - 1:
            if args.task_name in {'wikitext', 'wiki103', 'arxiv21'}:
                writetoxlsx(args.task_name, args.model_name, args.n_epochs,
                            args.forward_attack, args.forward_attack_rate,
                            args.backward_attack, args.backward_attack_rate,
                            args.optimizer,
                            result["perplexity_custom"]["perplexity"], result["perplexity_custom"]["loss"])
                if args.wandb:
                    wandb.config.result = result
    else:
        for i, data in enumerate(test_data_loader):
            pipeline.infer_iter(None, None, None)

def distributed_test_bert_iter(args, pipeline, device, test_data_loader):
    pipeline.model.eval() # Flag .training to True to enable Dropout
    if get_pipeline_parallel_rank() == 0:
        for i, data in enumerate(test_data_loader):
            inputs_ids = data['text'].to(device)
            aux_inputs = {
                'token_type_ids': data['token_type_ids'].to(device),
                'attention_mask': data['attention_mask'].to(device),
            }
            current_iter_time = pipeline.infer_iter(inputs_ids, aux_input_data=aux_inputs)
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        metrics = get_metric(args)
        for i, data in enumerate(test_data_loader):
            aux_inputs = {
                'attention_mask': data['attention_mask'].to(device),
            }
            input_ids = data['text'].to(device)
            labels = data['label'].to(device)
            pipeline.infer_iter(None, labels, aux_input_data=aux_inputs, metrics=metrics)
        
        result = {
            metric.name: metric.compute() for metric in metrics
        }
        print(result)
    else:
        for i, data in enumerate(test_data_loader):
            aux_inputs = {
                'attention_mask': data['attention_mask'].to(device),
            }
            pipeline.infer_iter(aux_input_data=aux_inputs)