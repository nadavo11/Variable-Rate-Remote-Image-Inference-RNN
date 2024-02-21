# VRIC-RNN
VARIABLE RATE IMAGE COMPRESSION WITH RNN - PYTORCH

# usage:
## use the cli to train and evaluate models
python .\train.py

- *example*:
python .\train.py --model conv --patch_size 32 --num_epochs 50 --batch_size 16 --log_step 3 --learning_rate 0.01 --coded_size 64

Compression for low-resolution thumbnails images using RNN.

Original paper: Toderici et al., Google, ICLR 2016 
