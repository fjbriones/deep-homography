Implementation of <a href="https://arxiv.org/abs/1606.03798">Deep Homography</a> in Keras

This repository implements Deep Homography in Keras. It contains both the code to generate dataset from any large image dataset (MSCOCO, CIFAR 10, etc.) and the code to train them. The trained models for the dataset MSCOCO 2014 are stored in the models directory.

Processed data can be downloaded <a href="https://1drv.ms/f/s!Ao8Y5FscWK9imoYp7eWyvlNfZMHIuA">here</a>

Uses of functions:

<i>In generate-dataset.py</i>
<ul>
  <li>load_random_image(path_source, size=128)</li>
  <ul>
    <li>path_source - directory where to get a random image of file format .jpg</li>
    <li>size - resize the image</li>
  </ul>
  <li>save_to_file(images, offsets, path_dest)</li>
  <ul>
    <li>images - array of image pairs to be stored</li>
    <li>offsets - offset relationship of the two images</li>
    <li>path_dest = path where to store the saved array as npz</li>
  </ul>
  <li>
  <li>generate_dataset(path_source, path_dest, rho, height, width, data, box)</li>
  <ul>
    <li>path_source - source of the images</li>
    <li>path_dest - path where to save the converted images as arrays</li>
    <li>rho - range of possible offsets</li>
    <li>height - height of the resized image</li>
    <li>width - width of the resized image</li>
    <li>data - number of image pairs to be generated</li>
    <li>box - size of the final image pair</li>
  </ul> 
</ul>


