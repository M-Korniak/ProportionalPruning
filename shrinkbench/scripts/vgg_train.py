import os
from shrinkbench.experiment import PruningExperiment

os.environ['DATAPATH'] = 'data/CIFAR10'

exp = TrainingExperiment(dataset='CIFAR10', 
                                model='VGG11',
                                pretrained=False,
                                train_kwargs={'epochs':2})
exp.run()