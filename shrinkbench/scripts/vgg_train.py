import os
from shrinkbench.experiment import TrainingExperiment

print("Hello from vgg_train")
os.environ['DATAPATH'] = '$HOME/ProportionalPruning/data/CIFAR10'
print("Datapath env set")
exp = TrainingExperiment(dataset='CIFAR10', 
                                model='vgg11',
                                pretrained=False,
                                train_kwargs={'epochs':2})
exp.run()