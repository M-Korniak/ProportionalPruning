import os
from shrinkbench.experiment import TrainingExperiment

os.environ['DATAPATH'] = '/ProportionalPruning/data/CIFAR10'

print(os.getcwd())
print(os.environ['DATAPATH'])

exp = TrainingExperiment(dataset='CIFAR10', 
                                model='vgg11',
                                pretrained=False,
                                train_kwargs={'epochs':2})
exp.run()