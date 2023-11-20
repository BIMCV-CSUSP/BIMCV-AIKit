# BIMCV code utils

## Training module usage

The code in this repo is an MNIST example of the template.
Try `bimcv_train -c config.json` to run code.

### Config file format

Config files are in `.json` format:

```javascript
{
  "name": "Mnist_LeNet",        // training session name
  "description": "",            // free text description of the experiment (optional)
  "task": "classification",     // used to save logs (i.e. one folder per task)
  "n_gpu": 1,                   // number of GPUs to use for training.
  "arch": {
    "module": "bimcv_aikit.models.classification", // name of module where the architecture is defined
    "type": "MnistModel",       // name of model architecture to train
    "args": {

    }                
  },
  "data_loader": {
    "type": "MnistDataLoader",         // selecting data loader
    "partitions": {
      "folds": [                       // number of folds for cross-validation
          "Fold 0",                    // see ADNI dataloader for an example
          "Fold 1",
          "Fold 2",
          "Fold 3",
          "Fold 4",
          ...
      ],
      "train": "train",                // name of the train partition to retrieve it from the dataloader
      "val": "dev",                    // name of the validation partition to retrieve it from the dataloader
      "test": "test"                   // name of the test partition to retrieve it from the dataloader
    },
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 64,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": {
    "module": "torch.nn",              // name of module where the loss function is defined
    "type": "CrossEntropyLoss",
    "args": {}
  },              
  "metrics": {
    "accuracy": {
      "module": "torchmetrics",        // name of module where the metric is defined
      "type": "Accuracy",
      "args": {
        "task": "multiclass",
        "average": "weighted",
        "num_classes": 10
      }
    },
    "specificity": {
      "module": "torchmetrics",
      "type": "Specificity",
      "args": {
        "task": "multiclass",
        "average": "weighted",
        "num_classes": 10
      }
    }
  },                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "type": "ClassificationTrainer",   // name of the trainer to use
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10                   // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

Add addional configurations if you need.

### Using config files

Modify the configurations in `.json` config files, then run:

```python
bimcv_train -c config.json
```
or 
```python
python -m bimcv_aikit.training.train -c config.json
```

## TODO

- [ x ] Enable training from pip package installation

## Acknowledgements

This project is inspired by [pytorch-template](https://github.com/victoresque/pytorch-template) the project by [Victor Huang](https://github.com/victoresque)
