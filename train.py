#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from utils.profiler import Profiler
import config
import torch
import datasets
from experiment import  training
import argparse,argcomplete
import transformation_measure as tm
from typing import Tuple
from transformation_measure.iterators.pytorch_image_dataset import TransformationStrategy

def list_parser(s:str):
    s=s.strip()
    if s=="":
        return []
    delim=","
    string_list=s.split(delim)
    return [int(n) for n in string_list]

def parse_args()->Tuple[training.Parameters, training.Options,float]:

    bool_parser=lambda x: (str(x).lower() in ['true','1', 'yes'])

    transformations=config.all_transformations()
    transformations={t.id():t for t in transformations}

    model_configs = config.all_models()
    parser = argparse.ArgumentParser(description="Script to train a models with a dataset and transformations")


    parser.add_argument('-max_restarts', metavar='mr'
                        , help=f'Maximum number of restarts to train the model until a minimum accuracy is reached'
                        , type=int
                        , default=4)
    parser.add_argument('-min_accuracy', metavar='macc'
                        , help=f'minimum accuracy required'
                        , type=float
                        , default=0.0)

    parser.add_argument('-batchsize', metavar='b'
                        , help=f'batchsize to use during training'
                        , type=int
                        , default=256)

    parser.add_argument('-num_workers', metavar='nw'
                        , help=f'num_workersto use during training'
                        , type=int
                        , default=2)

    parser.add_argument('-epochs', metavar='epo'
                        , help=f'Epochs to train the model'
                        ,required=True
                        , type=int)

    parser.add_argument('-notransform_epochs', metavar='nte'
                        , help=f'Train with no transformations for notransform_epochs epochs'
                        , type=int
                        , default=0)

    parser.add_argument('-savepoints', metavar='b'
                        , help=f'Epochs where models is to be saved'
                        , type=list_parser
                        , default=[])

    parser.add_argument('-plots', metavar='p'
                        , help=f'Generate plots for training epochs'
                        , type=bool_parser
                        , default=True)

    parser.add_argument('-model', metavar='m',
                        help=f'Model to train/use. Allowed values: {", ".join(model_configs.keys())}'
                        ,choices=model_configs.keys()
                        ,required=True)

    parser.add_argument('-usecuda', metavar='c'
                        , help=f'Use cuda'
                        , type=bool_parser
                        , default=torch.cuda.is_available())

    parser.add_argument('-dataset', metavar='d',
                        help=f'Dataset to train/eval models. Allowed values: {", ".join(datasets.names)}'
                        ,choices=datasets.names
                        ,required=True)

    parser.add_argument('-transformation', metavar='t',
                        help=f'Transformations to apply to the dataset to train a models. Allowed values: {", ".join(transformations.keys())}'
                        , choices=transformations.keys()
                        ,default=tm.SimpleAffineTransformationGenerator())

    parser.add_argument('-suffix', metavar='suff'
                        , help=f'Suffix to add to model name'
                        , type=str
                        , default="")

    parser.add_argument('-verbose',
                        help=f'Print info about dataset/models/transformations',
                        action="store_true", )

    parser.add_argument('-train_verbose',
                        help=f'Print details about the training',
                        action="store_true", )

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    transformation=transformations[args.transformation]
    model_config = model_configs[args.model]
    p= training.Parameters(model_config, args.dataset, transformation, args.epochs, args.notransform_epochs,args.savepoints,args.suffix)

    o= training.Options(args.verbose, args.train_verbose, True, args.batchsize, args.num_workers, args.usecuda, args.plots,args.max_restarts)

    min_accuracy = args.min_accuracy
    return p,o,min_accuracy


if __name__ == "__main__":
    p,o,min_accuracy = parse_args()
    def do_train():
        dataset = datasets.get(p.dataset)
        model,optimizer = p.model.make_model(dataset.input_shape, dataset.num_classes, o.use_cuda)
        p.transformations.set_input_shape(dataset.input_shape)
        # p.transformations.set_pytorch(True)
        # p.transformations.set_cuda(o.use_cuda)
        # an="\n".join(model.activation_names())
        # print("Activation names: "+an)
        if o.verbose:
            print("Parameters: ",p)
            print("Options: ",o)
            print("Min accuracy: ",min_accuracy)

        def generate_epochs_callbacks():
            epochs_callbacks=[]
            for epoch in p.savepoints:
                def callback(epoch=epoch):
                    scores=training.eval_scores(model,dataset,p.transformations,TransformationStrategy.random_sample,o.get_eval_options())
                    if o.verbose:
                        print(f"Saving model {model.name} at epoch {epoch}/{p.epochs}.")
                    training.save_model(p, o, model, scores, config.model_path(p, epoch))
                epochs_callbacks.append((epoch,callback))

            return dict(epochs_callbacks)

        epochs_callbacks=generate_epochs_callbacks()

        if o.verbose:
            print(f"Experimenting with dataset {p.dataset}.")
            print(dataset.summary())
            print(f"Training with models {p.model}.")
            print(model)
            if len(p.savepoints):
                epochs_str= ", ".join([ str(sp) for sp in p.savepoints])
                print(f"Savepoints at epochs {epochs_str}.")



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




