# Final-Project-ECE-176

This project trains a small CNN on CIFAR-10 and evaluates how performance changes under image corruptions. These corruptions can be gaussian noise, blur, and brightness shifts. 

## First Part 
- Train a baseline CNN on the clean Dataset CIFAR-10
- Evaluate clean test accuracy
- Save the best model

## File Structure
 - `config.py` - training and system configuration for tweaking i.e epoch and learning rates. 
 - `dataset.py` - loads the dataset (CIFAR-10) 
 - `models/cnn.py` - CNN model
 - `utils/training_utils.py` - training and evaluation functions
 - `train.py` - main training script

## Running the program
```bash
python train.py
```

## still need to implement 
- corruption (blur, brightness, shifts, wtv)
- evaluation experiments (testing accuracy)
- plotting results

## optional 
- as suggested by the TA we can implement a harder/larger dataset
  
