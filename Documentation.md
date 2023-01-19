# Documentation
## Usage
### Specify Config File Explicitly
Usage: `python main.py config_file`     

The argument "config_file" is the path to the config file.

### Fall Back
If the config file does not exist or you don't specify the config_file (i.e., execute `python main.py`), the program will fall back to execute all the config files (except "minist.yaml") parallelly using multiprocessing. 

### Help
Using `python main.py -h` to get the help.

## How Did I Implement It
### Training Function
The main training part comes from the template. What I have implemented is the recording and outputing function.     

I defined two variables "train_acc" and "train_loss" to record the accumulative training accuracy and training loss and initialize them to 0. For each batch, the loss.item() is added into train_loss and 1 is added into train_acc if the output matches the target, or else it will add 0. At last we let them divide by the length of the dataset (train_loader.dataset) to get the average training accuracy and training loss. Use "print" function and file I/O to output the result to both standard I/O and text file.

### Testing Function
I defined two variables "test_loss" and "correct" to record the count of correct testings and the accumulative training loss and initialize them to 0. For each test, loss item is added into test_loss. 1 is added into correct if the prediction matches the target, or else it will add 0. At last we let them divide by the length of the dataset to get the average testing accuracy and testing loss. Use "print" function and file I/O and text file. 

### Plotting Function
The plotting function uses "matplotlib" to draw the plot. The plotting function has three parameters. The first one is the epochs list. This is actually a finite sequence from 1 to epochs. The second one is the performance. This is a list containing the training data. The third one is the title. Then I use matplotlib.pyplot to draw the graph. Along with the graph are the labels and the title.

### Ramdom Seed
Three copies of the default config file was created, each modified it's seed. The seeds of the thee copies are set as 123, 321, 666 as required. If the parameter of the program is not set, it will automatically fall back to iterating all the config files in "config" directory. 

### Python Multiprocessing
Multiprocessing is implemented by "multiprocessing" module. The "Process" function is used to execute the "run" function parallelly. The "Pipe" function is used to transfer the training and testing data to the outside. 