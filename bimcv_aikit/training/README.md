# BIMCV code utils

## Training module usage

This guide explains how to configure your deep learning model training using JSON configuration files. The configuration system allows you to set up various aspects of your model, data loading, training process, and evaluation metrics without modifying the core code.

### Configuration Structure

The configuration file is a JSON document with several main sections:

- General Settings
- Model Architecture
- Data Loading
- Optimizer
- Loss Function
- Metrics
- Learning Rate Scheduler
- Trainer
- Post-processing Transforms
- Inference

Let's break down each section and explain the configurable fields:

1. General Settings

`name`: A string identifying your experiment (e.g., "Mnist_LeNet", "OxfordPet_UNet").
`task`: The type of task, either "classification" or "segmentation". At the end this will be used to name one level of the output directory, so it ends up organized by task type.
`n_gpu`: Number of GPUs to use (e.g., 0 for CPU, 1 for single GPU).
`seed`: Random seed for reproducibility.

2. Model Architecture
The arch field specifies the model architecture:

```json
"arch": {
  "module": "path.to.module",
  "type": "ModelClassName",
  "args": {}
}
```

`module`: The Python module containing the model class.
`type`: The name of the model class.
`args`: Any arguments to pass to the model constructor.

3. Data Loading
The data_loader field configures how data is loaded and preprocessed:

```json
"data_loader": {
  "module": "path.to.dataloader.module",
  "type": "DataLoaderClassName",
  "partitions": {
    "train": "train",
    "val": "dev",
    "test": "test"
  },
  "args": {
    "data_dir": "path/to/data/",
    "batch_size": 32,
    "shuffle": true,
    "num_workers": 4,
    "transforms": {
      "train": { ... },
      "val": { ... },
      "test": { ... }
    }
  }
}
```

`module` and `type`: Specify the data loader class.
`partitions`: Define dataset splits. In the example above, "dev" will be used to query the dataloader for the validation split. If you don't have a validation or test set, the dataloader must return `None` for the specified partition.
`args`: Configure batch size, shuffling, number of workers, and data transforms.

Transforms can be specified for each partition (train, val, test) and for both input data and labels (in segmentation tasks). The keys on the transform dictionary in the config file must match those used in the dataloader.

4. Optimizer
Choose and configure the optimization algorithm:

```json
"optimizer": {
  "type": "Adam",
  "args": {
    "lr": 0.001,
    "weight_decay": 0,
    "amsgrad": true
  }
}
```

`type`: The optimizer class name (e.g., "Adam", "SGD"). At this point, it must be from the `torch.optim` module.
`args`: Optimizer-specific parameters.

5. Loss Function
Specify the loss function:

```json
"loss": {
  "module": "torch.nn",
  "type": "CrossEntropyLoss",
  "args": {}
}
```

`module`: The module containing the loss function.
`type`: The name of the loss function class.
`args`: Any arguments for the loss function.

6. Metrics
Define evaluation metrics:

```json
"metrics": {
  "accuracy": {
    "module": "torchmetrics.functional.classification",
    "type": "accuracy",
    "args": {
      "task": "multiclass",
      "average": "weighted",
      "num_classes": 10
    }
  }
}
```

You can specify multiple metrics, each with its own configuration.

7. Learning Rate Scheduler
Optionally configure a learning rate scheduler:

```json
"lr_scheduler": {
  "type": "StepLR",
  "args": {
    "step_size": 50,
    "gamma": 0.1
  }
}
```

8. Trainer
Configure the training process:

```json
"trainer": {
  "type": "ClassificationTrainer",
  "epochs": 5,
  "save_dir": "saved/",
  "save_period": null,
  "verbosity": 2,
  "monitor": "min loss",
  "early_stop": 10,
  "tensorboard": false
}
```

This section controls training duration, model saving, logging, early stopping, and TensorBoard integration.

9. Post-processing Transforms
Define any post-processing steps applied to model outputs:

```json
"post_transforms": {
  "pred": {
    "module": "monai.transforms",
    "type": "Compose",
    "args": {
      "transforms": [
        {
          "module": "monai.transforms",
          "type": "Activations",
          "args": {
            "softmax": true
          }
        },
        {
          "module": "monai.transforms",
          "type": "AsDiscrete",
          "args": {
            "argmax": true,
            "dim": 1,
            "keepdim": true
          }
        }
      ]
    }
  }
}
```

10. Inference
Configure the inference process:

```json
"inferer": {
  "module": "monai.inferers",
  "type": "SimpleInferer",
  "args": {}
}
```

### Task-Specific Configurations

#### Classification

For classification tasks, pay special attention to:

- The model architecture (often a CNN-based model)
- Classification-specific metrics (e.g., accuracy, precision, recall)
- The number of classes in your dataset

#### Segmentation

For segmentation tasks, consider:

- Using a segmentation-specific architecture (e.g., U-Net)
- Segmentation-specific loss functions (e.g., Dice Loss)
- Metrics suitable for segmentation evaluation (e.g., Mean Dice score)
- Proper input and label transformations, including resizing

### Customization

You can adjust the configuration as needed. In particular, pay attention to the dataloader section, which is usually customized for each use case. Remember to adjust paths, class names, and specific parameters to match your project structure and requirements. See the examples configurations for [classification](./config_classification.json) and [segmentation](./config_segmentation.json) for more details.

## Acknowledgements

This project is inspired by [pytorch-template](https://github.com/victoresque/pytorch-template) the project by [Victor Huang](https://github.com/victoresque).
