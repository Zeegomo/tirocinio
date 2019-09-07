# tirocinio

## Requirements:
C++11 compiler

## General 
### `parameters.json` config
* `input_dim`: n. of input variables
* `output_dim`: n. of output variables (1)
* `hidden_dim`: n. of neurons in the hidden layer
* `num_epochs`: n. of training epochs
* `batch_size`: size of each training sample
* `traning_samples`: n. of training samples
* `learning_rate`: learning rate,
* `target_column`: the column to predict (calculated after ignore columns)
* `ignore`: columns to be ignored

### Usage
Dataset will be read from stdin.

If you launch the program without additional parameters it trains on a subset of the dataset (specified in `parameters.json`) and then tests the trained model on the whole sequence.
The results of the evaluation task is available in `out.data` file, where the first line is the original data and the second is the predicted one. 

You can use `print.py` (requires matplotlib) to display those data.

### Save and load model parameters (only available in Pytorch)
If you want to save or load the trained model you have to specify a path to an existing directory where the parameters will be saved.
You can do that at launch:
* `py-rnn MODE YOUR_LOCATION < YOUR_DATASET`

`MODE` specifies what the program will do:

* `t` : train the model from scratch and save to `YOUR_LOCATION`
* `e` : load model from `YOUR_LOCATION` and test it on `YOUR_DATASET`. Results will be saved in `out.data`. 
* `te`, `te`: train the model  from scratch and test it on `YOUR_DATASET`. Model will be save in `YOUR_LOCATION` and results in `out.data` .

`YOUR_LOCATION` is the path to an existing directory.

## Windows
* Clone repo
### Pytorch
1) go to pytorch\pytorch_lib\windows\ and unzip the file (you should have pytorch\pytorch_lib\windows\libtorch, if not go inside the unzipped directory and copy the libtorch folder in the parent directory)
2) Open `pytorch/CMakeLists.txt` with Visual Studio
3) Wait for VS to generate CMake Cache then CMake->Build All
4) Copy all the .dll files in libtorch\ in the directory where the executable is located (Users\CMakeBuilds for VS 2017, tirocinio\pytorch\build\.. for VS 2019).
5) Copy (and edit) sample `parameters.json` in the directory where the executable is located.
6) Run the executable. Training data must be passed from stdin.
### ANNT
* Steps 2, 3, 5, 6 but open `annt/CMakeLists.txt` instead of `pytorch/CMakeLists.txt`.
## Unix

### Pytorch
1) Clone repo
2) Download and unzip [Libtorch](https://pytorch.org/) inside `tirocinio/pyorch_lib/unix/` if not present.
3) `cd tirocinio/pytorch`
4) `mkdir build`
5) `cd build`
6) run `cmake ..`
7) Steps 5, 6 from Windows Pytorch

### ANNT
1) Clone repo
2) Download and build [ANNT](https://github.com/cvsandbox/ANNT) inside `tirocinio/annt_lib/unix/` if not present.
7) Steps 3, 4, 5, 6 from Unix Pytorch.
