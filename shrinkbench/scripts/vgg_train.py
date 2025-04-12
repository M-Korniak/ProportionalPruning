import os
from shrinkbench.experiment import TrainingExperiment

os.environ['DATAPATH'] = '$HOME/ProportionalPruning/data/CIFAR10'

exp = TrainingExperiment(dataset='CIFAR10', 
                                model='vgg11',
                                pretrained=False,
                                train_kwargs={'epochs':2})
exp.run()