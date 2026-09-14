# Postman AI/ML Task — Tasks 1 : Neural Network from scratch - README file

## Requirements
Python 3.10+ · numpy · matplotlib · pandas · torch (reference)

## Installation
    pip install -r requirements.txt

## Running the code
    Every thing is vey simplified so that the code runs smoothly.
    Just make sure MNIST csv file is in same folder as the code file. 
    I Recommend to run this code on terminal
    After first run, a `trained_weights.npz` file will be created and after every run, it will be modified to increase accuracy of the network



## Results
1.1 Build a small neural network - **Complete**

1.2 Implement the forward pass using linear layers and an activation function of your choice. - **Complete**

1.3 Implement the backward pass manually using the chain rule. Derive the gradients for every
layer and include the mathematics in your write-up or code comments. - **Complete**

1.4 Check your gradients against torch.autograd or numerical gradient checking. Your results
should match within a reasonable tolerance. - **Complete** (delta between the matrices are in range $$10^{-6}$$ to $$10^{-8}$$ which are acceptable)

1.5 Train the network on a small real task, such as a reduced MNIST dataset or a simple classifi-
cation or regression dataset. Show that the loss decreases during training. - **Complete** (Using MNIST dataset)

1.6 In your write-up, discuss any gradient mistakes you made and explain how you identified and
fixed them. - **Complete**

## References
*Neural Networks and Deep Learning* by Nielsen
3B1B
StackOverflow
Gemini, ChatGPT
References for code implementation are given within code in comments

