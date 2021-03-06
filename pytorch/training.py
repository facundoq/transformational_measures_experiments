import progressbar
import torch,torch.multiprocessing
import numpy as np
import transformational_measures as tm
from typing import Callable
def print_results(dataset,loss,accuracy,correct,n):
    print('{} => Loss: {:.4f}, Accuracy: {:.2f}% ({}/{})'.format(dataset,
        loss, 100. * accuracy, correct, n),flush=True)

def train(model:torch.nn.Module,epochs:int,optimizer,use_cuda:bool,train_dataset,test_dataset,loss_function,verbose=True,max_epochs_without_improvement_p=0.1,max_epochs_without_improvement_treshold=1e-3,eval_test_every_n_epochs:int=None,epochs_callbacks:{int:Callable}={}):

    if eval_test_every_n_epochs == None:
        eval_test_every_n_epochs= max(epochs//10,1)

    # torch.multiprocessing.set_start_method("spawn")
    history={"acc":[],"acc_val":[],"loss":[],"loss_val":[]}
    max_epochs_without_improvement=max(int(max_epochs_without_improvement_p*epochs),1)

    model.train()

    last_accuracy=0
    no_improvement_epochs=0

    test_results=(0,0)

    for epoch in range(1, epochs + 1):
        loss,accuracy,correct,n=train_epoch(model,epoch,optimizer,use_cuda,train_dataset,loss_function,verbose)

        if epoch in epochs_callbacks:
            epochs_callbacks[epoch]()

        if epoch == 0 or epoch==epochs or epoch % eval_test_every_n_epochs == 0:
            test_results = test(model,test_dataset,use_cuda,loss_function)
            if verbose:
                print_results("Test", *test_results)
        history["loss"].append(loss)
        history["loss_val"].append(test_results[0])
        history["acc"].append(accuracy)
        history["acc_val"].append(test_results[1])

        # abort if no improvement in various epochs
        if abs(last_accuracy-accuracy)<max_epochs_without_improvement_treshold:
            no_improvement_epochs += 1
        else:
            no_improvement_epochs =0
        if no_improvement_epochs==max_epochs_without_improvement:
            if verbose:
                print(f"Stopping training early, epoch {epoch}/{epochs}, epochs without improvement= {max_epochs_without_improvement_treshold}")
            break

    return history


def test(model, dataset, use_cuda,loss_function):
    with torch.no_grad():
        model.eval()
        loss = 0
        correct = 0

        for data, target in dataset:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            loss += loss_function(output,target[:,0]).item()*data.shape[0]
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    n=len(dataset.dataset)
    loss /= n
    accuracy = float(correct) / float(n)
    return loss,accuracy,correct,n

def train_epoch(model,epoch,optimizer,use_cuda,train_dataset,loss_function,verbose):
    n=len(train_dataset)
    update_every_n_batches= max(n // 5,1)
    if verbose:
        widgets = ["Epoch {}: ".format(epoch), progressbar.Percentage()
                   ,progressbar.FormatLabel(' (batch %(value)d/%(max_value)d) ')
                   ,' ==stats==> ', progressbar.DynamicMessage("loss")
                   ,', ',progressbar.DynamicMessage("accuracy")
                   ,', ',progressbar.ETA()
                   ]
        progress_bar = progressbar.ProgressBar(widgets=widgets, max_value=len(train_dataset)).start()
    batches=len(train_dataset)
    losses=np.zeros(batches)
    accuracies=np.zeros(batches)
    correct=0

    for batch_idx, (data, target) in enumerate(train_dataset):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        #MODEL OUTPUT
        output = model(data)
        loss = loss_function(output, target[:,0])
        # UPDATE PARAMETERS
        loss.backward()
        optimizer.step()


        # ESTIMATE BATCH LOSS AND ACCURACY
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        matches = pred.eq(target.data.view_as(pred)).cpu()
        correct += matches.sum()
        accuracies[batch_idx] = matches.float().mean().item()
        losses[batch_idx] = loss.cpu().item()

        # UPDATE UI
        if (batch_idx % update_every_n_batches == 0 or batch_idx+1 == n) and verbose:
            progress_bar.update(batch_idx+1,loss=losses[:batch_idx+1].mean(),accuracy=accuracies[:batch_idx+1].mean())
    if verbose:
        progress_bar.finish()
    return losses.mean(),accuracies.mean(),correct,len(train_dataset.dataset)


def eval_scores(models,datasets,config,loss_function):
    scores = {}
    for model_name in sorted(models):
        m = models[model_name]
        for dataset_name in sorted(datasets):
            dataset = datasets[dataset_name]
            key = model_name + '_' + dataset_name
            #print(f"Evaluating {key}:")
            loss,accuracy,correct,n=test(m,dataset,config.use_cuda,loss_function)

            scores[key] = (loss,accuracy)

    return scores



def add_weight_decay(parameters, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in parameters:
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]
