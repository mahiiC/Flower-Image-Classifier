# Flower Image Classifier Project
#### Udacity-AWS AI Programming with Python Nanodegree
In this project, we delve into the exciting world of deep learning to tackle a specific task: classifying flowers. The project has been completed using Udacity's GPU-enabled workspace and it uses transfer learning to build a robust and efficient flower classification system using an image dataset having 102 flower categories.  
Users are provided with the option to choose from three pre-trained models from PyTorch's Torchvision package- VGG19, AlexNet and DenseNet121  
A new, untrained, feed-forward network architecture was defined as a classifier using ReLU activation functions. Dropout function was used to prevent overfitting of the model to the data.  
The code developed for the image classifier was then converted to command line applications train.py and predict.py in part 2 of the project.

---

### Model training
The application recognizes 102 different species of flowers. The original dataset can be found here: [http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html]. All the images are in the flowers folder. In the training set there are more than 6000 images of flowers. Considering the relatively small sample size, transfer learning has been used in which a pretrained neural network model (vgg19, densenet121, and alexnet) have been borrowed, and only the last several layers will be trained on the flower training set to fine-tune the weights. ReLU activation and Softmax output are used. Dropout method is used to avoid over-fitting.

During the training process, the model was also tested on the validation dataset, and the loss and accuracy were displayed. After the model was trained, it was applied on the test dataset and achieved an accuracy rate of around 80%. Images can also be provided by user for the prediction of flower species. The top k (for example top 5) classes will be returned with corresponding probabilities.

GPU is recommended to use for the training process, and the estimated training time using GPU is around 30-60 minutes. Once training is done, the model will be saved. Using the saved model, prediction can be made very fast.

### Notebook
Please refer to the Jupyter Notebook file Image Classifier Project (4) (1) (1).ipynb for the implementation of the model training and prediction. he trained model was saved as checkpoint.pth, which can be loaded to make quick prediction on any input imagesThe VGG19 model was used as the pre-trained transfer learning model. The input image and prediction result are nicely displayed in a plot. For example, for a flower image randomly downloaded from internet, this model successfully predicted its category. 

### Command line scripts
For command line applications, there are two scripts: train.py and predict.py. Running train.py will train the model using the training dataset; on the other hand, the predict.py will load the trained model and make prediction for input image. 

---

### train.py
This file (`train.py`) is a PyTorch script designed to train a deep learning model for flower species classification. Here's a breakdown of its functionality:

**1. Imports and Setup:**

-   Imports necessary libraries like `torch`, `torchvision`, `matplotlib`, `numpy`, `pandas`, and `argparse`.
-   Loads a JSON file (`cat_to_name.json`) that maps category IDs to flower names.
-   Defines data directories (`train_dir`, `valid_dir`, `test_dir`).

**2. Data Transformations and Loading:**

-   Defines data transformations (`train_transforms`, `valid_transforms`, `test_transforms`) for data augmentation and normalization.
-   Loads the image datasets using `datasets.ImageFolder` and applies the defined transformations.
-   Creates data loaders (`trainloader`, `validloader`, `testloader`) for batching and shuffling the data.

**3. Model Definition (`classifier` function):**

-   This function defines the model architecture.
-   It allows selection from pre-trained models: `vgg19`, `resnet18`, or `alexnet`.
-   It freezes the pre-trained model's parameters to prevent them from being updated during training (transfer learning).
-   It adds a custom classifier (fully connected layers) on top of the pre-trained model.
-   The classifier includes dropout for regularization and a final Softmax layer for classification.
-   It takes `hidden_layers` and `dropout` as input parameters.

**4. Argument Parsing:**

-   Uses `argparse` to handle command-line arguments:
    -   `--model`: Specifies the pre-trained model to use.
    -   `--hidden_layers`: Sets the number of hidden units in the classifier.
    -   `--dropout`: Sets the dropout probability.
    -   `--data_dir`: Specifies the data directory.
    -   `--checkpoint`: Specifies the path to save the trained model.
    -   `--learning_rate`: Sets the learning rate for the optimizer.
    -   `--epochs`: Sets the number of training epochs.
    -   `--gpu`: determines if a gpu is available.
-   Parses these arguments and stores them in the `args` variable.

**5. Model Training (`training` function):**

-   Trains the model using the training data and validates it using the validation data.
-   Iterates through epochs and batches of data.
-   Calculates loss and performs backpropagation.
-   Prints training and validation loss and accuracy.
-   Utilizes GPU if available.

**6. Model Testing (`test_model` function):**

-   Evaluates the trained model's performance on the test dataset.
-   Calculates and prints the test accuracy.

**7. Model Saving (Checkpoint):**

-   Saves the trained model's state, along with other relevant information (architecture, learning rate, etc.), to a checkpoint file. This allows you to load and use the trained model later.

---

### predict.py
This file (`predict.py`) is a PyTorch script designed to load a trained model and predict the class of a given flower image. Here's a breakdown of its functionality:

**1. Imports and Setup:**

-   Imports necessary libraries like `torch`, `torchvision`, `matplotlib`, `numpy`, `PIL`, `json`, and `argparse`.
-   Uses `argparse` to handle command-line arguments:
    -   `--testfile`: Specifies the path to the image file to be classified.
    -   `--jsonfile`: Specifies the path to the JSON file containing category-to-name mappings.
    -   `--checkpointfile`: Specifies the path to the saved model checkpoint.
    -   `--topk`: Specifies the number of top predicted classes to return.
-   Parses these arguments and stores them in the `args` variable.
-   Loads the category-to-name mapping from the JSON file.

**2. Loading the Checkpoint (`load_checkpoint` function):**

-   Loads the saved model checkpoint from the specified file.
-   Extracts model architecture, learning rate, dropout, output size, hidden layers, state dictionary, epochs, and class-to-index mapping from the checkpoint.
-   Recreates the model architecture based on the saved information (VGG19, ResNet18, or AlexNet).
-   Loads the saved model state (`state_dict`) into the model.
-   Freezes the model's parameters to prevent them from being updated.
-   Returns the loaded model.

**3. Image Preprocessing (`process_image` function):**

-   Takes an image path as input and preprocesses the image for use in the PyTorch model.
-   Opens the image using PIL (`Image.open`).
-   Applies transformations:
    -   Resizes the image.
    -   Center crops the image.
    -   Converts the image to a PyTorch tensor.
    -   Normalizes the tensor using predefined mean and standard deviation.
-   Returns the preprocessed image tensor.

**4. Prediction (`predict` function):**

-   Takes the image path, loaded model, and `topk` as input.
-   Processes the image using the `process_image` function.
-   Adds a batch dimension to the image tensor and moves it to the appropriate device (GPU if available, CPU otherwise).
-   Performs forward pass through the model to obtain predictions.
-   Applies Softmax to get probabilities.
-   Gets the top `topk` probabilities and their corresponding indices.
-   Converts the indices to class labels using the class-to-index mapping from the loaded model.
-   Returns the top probabilities and class labels.

**5. Prediction Execution:**

-   Loads the model using the `load_checkpoint` function.
-   Calls the `predict` function to get the top `topk` predictions for the specified image.
-   Prints the top probabilities and class labels.



