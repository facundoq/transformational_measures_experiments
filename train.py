#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from utils.profiler import Profiler
import config
import torch
import datasets
from experiment import  training
from pytorch.pytorch_image_dataset import TransformationStrategy


def main(p:training.Parameters,o:training.Options,min_accuracy:float):

    dataset = datasets.get(p.dataset)
    dataset.normalize_features()

    # p.transformations.set_input_shape(dataset.input_shape)
    # p.transformations.set_pytorch(True)
    if o.verbose_general:
        print("Parameters: ",p)
        print("Options: ",o)
        print("Min accuracy: ",min_accuracy)
        print(f"Dataset {p.dataset}.")
        print(dataset.summary())
        print(f"Model {p.model}.")

        if len(p.savepoints):
            epochs_str= ", ".join([ str(sp) for sp in p.savepoints])
            print(f"Savepoints at epochs {epochs_str}.")

    def do_train():
        model,optimizer = p.model.make_model_and_optimizer(dataset.input_shape, dataset.num_classes, o.use_cuda)
        def generate_epochs_callbacks():
            epochs_callbacks=[]
            for epoch in p.savepoints:
                def callback(epoch=epoch):
                    scores=training.eval_scores(model,dataset,p.transformations,TransformationStrategy.random_sample,o.get_eval_options())
                    if o.verbose_general:
                        print(f"Saving model {model.name} at epoch {epoch}/{p.epochs}.")
                    training.save_model(p, o, model, scores, config.model_path(p, epoch))
                epochs_callbacks.append((epoch,callback))

            return dict(epochs_callbacks)

        epochs_callbacks=generate_epochs_callbacks()


    # TRAINING
        if 0 in p.savepoints:
            scores = training.eval_scores(model, dataset, p.transformations,TransformationStrategy.random_sample, o.get_eval_options())
            print(f"Saving model {model.name} at epoch {0} (before training).")
            training.save_model(p, o, model, scores, config.model_path(p, 0))
        pr = Profiler()
        pr.event("start")
        scores,history= training.run(p, o, model, optimizer, dataset,epochs_callbacks=epochs_callbacks)
        pr.event("end")
        print(pr.summary(human=True))

        training.print_scores(scores)
        return model,history,scores

    converged=False
    restarts=0


    test_accuracy=0
    model,history,scores=None,None,None
    while not converged and restarts<=o.max_restarts:
        if restarts > 0:
            message =f"""Model did not converge since it did not reach minimum accuracy ({test_accuracy}<{min_accuracy}). Restarting.. {restarts}/{o.max_restarts}"""
            print(message)
        model,history,scores=do_train()
        training.plot_history(history, p, config.training_plots_path())
        test_accuracy = scores["test"][1]
        converged= test_accuracy > min_accuracy
        restarts += 1

    # SAVING
    if o.save_model:
        if converged:
            path=config.model_path(p)
            training.save_model(p, o, model, scores, path)
            print(f"Model saved to {path}")
        else:
            print(f"Model was not saved since it did not reach minimum accuracy. Accuracy={test_accuracy}<{min_accuracy}.")
    # delete model and empty cuda cache

    del model
    del dataset
    torch.cuda.empty_cache()





