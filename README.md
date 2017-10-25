Implementation of <a href="https://arxiv.org/abs/1606.03798">Deep Homography</a> in Keras

This repository implements Deep Homography in Keras. It contains both the code to generate dataset from any large image dataset (MSCOCO, CIFAR 10, etc.) and the code to train them. The trained models for the dataset MSCOCO 2014 are stored in the models directory.

Processed data can be downloaded <a href="https://1drv.ms/f/s!Ao8Y5FscWK9imoYp7eWyvlNfZMHIuA">here</a>

Uses of functions:

<i>In generate-dataset.py</i>
<ul>
  <li>load_random_image(path_source, size=128) - loads a random image from a directory</li>
  <ul>
    <li>path_source - directory where to get a random image of file format .jpg</li>
    <li>size - resize the image</li>
  </ul>
  <li>save_to_file(images, offsets, path_dest) - saves an image pair in npz format</li>
  <ul>
    <li>images - array of image pairs to be stored</li>
    <li>offsets - offset relationship of the two images</li>
    <li>path_dest = path where to store the saved array as npz</li>
  </ul>
  <li>generate_dataset(path_source, path_dest, rho, height, width, data, box) - generate a dataset from a large group of images</li>
  <ul>
    <li>path_source - source of the images</li>
    <li>path_dest - path where to save the converted images as arrays</li>
    <li>rho - range of possible offsets</li>
    <li>height - height of the resized image</li>
    <li>width - width of the resized image</li>
    <li>data - number of image pairs to be generated</li>
    <li>box - size of the final image pair</li>
  </ul> 
  <li>group_dataset (path, new_path, box=128, size=64) - group a dataset by batches. This is generally recommended since training is significantly slower when the data is ungrouped</li>
  <ul>
    <li>path - path of the ungrouped dataset</li>
    <li>new_path - where to save the grouped dataset</li>
    <li>box - size of the image box</li>
    <li>size - size of grouped dataset. Default is at 64 which is the batch size in training</li>
  </ul>
</ul>

<i>In deep-homography.py</i>
<ul>
  <li>data_loader(path) - loads the npz files</li>
  <ul>
    <li>path - path where to load the data</li>
  </ul>
  <li>common(model) - the common architecture to both models</li>
  <ul>
    <li>model - model to be used for the network</li>
  </ul>
  <li>regression_network_last(model, lr) - the last layer of the regression network</li>
  <ul>
    <li>model - model to be used for the nework</li>
    <li>lr - learning rate of the model</li>
  </ul>
  <li>classification_network_last(model, lr, batch_size=64) - the last layer of the classification network</li>
  <ul>
    <li>model - model to be used for the nework</li>
    <li>lr - learning rate of the model</li>
    <li>batch_size - batch size of training</li>
  </ul>
  <li>regression_network(train_data_path, val_data_path, total_iterations, batch_size, itr, steps_per_epoch, val_steps) - training the entire regression network</li>
  <ul>
    <li>train_data_path - path of the data for training in npz format</li>
    <li>val_data_path - path of the data for validation in npz format</li>
    <li>total_iterations - total iterations for training</li>
    <li>batch_size - batch_size of training</li>
    <li>itr - number of iterations before changing the learning rate</li>
    <li>steps_per_epoch - number of steps per epoch for training</li>
    <li>val_steps - number of steps for validation</li>
  </ul>
  <li>classification_network(train_data_path, val_data_path, total_iterations, batch_size, itr, steps_per_epoch, val_steps) - training the entire classification network</li>
  <ul>
    <li>train_data_path - path of the data for training in npz format. Ensure that the data are grouped by batches. If not use the group_dataset function from generate-dataset.py</li>
    <li>val_data_path - path of the data for validation in npz format. Ensure that the data are grouped by batches. If not use the group_dataset function from generate-dataset.py</li>
    <li>total_iterations - total iterations for training</li>
    <li>batch_size - batch_size of training</li>
    <li>itr - number of iterations before changing the learning rate</li>
    <li>steps_per_epoch - number of steps per epoch for training</li>
    <li>val_steps - number of steps for validation</li>
  </ul>
  <li>test_model(model_save_path, test_data_path, test_size=5000, batch_size=64) - testing a saved model</li>
  <ul>
    <li>model_save_path - path where the models are saved</li>
    <li>test_data_path - path of the data for testing in npz format</li>
    <li>test_size - size of the testing data</li>
    <li>batch_size - batch size to be accepted by the network. Same as batch_size during training</li>
  </ul>
</ul>

<b>Current Performance</b>
<ul>
  <li>Regression Network MACE - 49.425491601</li>
  <li>Classification Network MACE - 71.5220545317</li>
</ul>
  
