# Postman Club Recruitment Writeup
## Day 1 - Saturday, September 12

> I learnt about neural networks by watching videos of 3b1b and got the basic mathematical idea of its calculas.
> Now, for deeper reference on mathematics and the network's algorithm, I referred to  **Neural Networks and Deep Learning** by *Michael Nielsen*
> This made me understand the maths quite lucidly.

> Now, the issue was the code itself. I have moderately learnt Python and NumPy, but this project required a deeper knowledge of them.
> I brushed up on some known topics like **OOP** and **NumPy shapes** to be able to code the next day efficintly.
> I want to manually write as much code as I can rather than vibe coding, as I would recognise and understand my own code better, and would be able to explain it better if I reach PI.

> So now, I wrote some test code on matrix operations to get hands-on experience with shapes and NumPy
> I didn't know how to import a CSV into my code, so I browsed the pandas library

## Day 2 - Sunday, September 13
> I actually started writing the code today. Also, I didn't understand git and github much, so I learnt about them through youtube and chatgpt

> ### The Algorithm for Neural Netwrok v0
>
>> This version will be simple, as I am new to all this and also due due to the time constraint I am facing. It will contain only 1 hidden layer and will have this structure:
>>
>> Input Layer -----> Hidden layer (64 neurons) ----> Output layer
>>
>> This would result in a not-so-very efficient network with success rate of about 0.85 to 0.9 according to Gemini's prediction
>
> 
>### Problems faced
> > Could not implement the one-hot function by myself. Fixed.
> > Code becomes very complex and messy very quickly; difficulty in keeping track of all variables and function parameters. Still not sure about them all, as the network is yet to be tested
> > Forgot backprop formulas. Revisited the handwritten notes I made yesterday. Fixed.
> > Still not very comfortable with shapes; have to think about them while writing every operation. **NOT FIXED**


## Day 3 - Monday, September 14

> ### Maths behind backprop
> Before starting with today's progress, let me present you with the math behind the curtains
> We will start by deriving results for a simple network, and intuitively apply that to a complex network
> > #### For a simple scenario
> > Consider a neural network as : Input layer (1 parameter) ----> Hidden layer (1 neuron) ----> Output layer (1 neuron)
> >
> > Using chain rule, $$\frac{\partial L}{\partial w^{[2]}} = \frac{\partial L}{\partial a^{[2]}}.\frac{\partial a^{[2]}}{\partial z^{[2]}}.\frac{\partial z^{[2]}}{\partial w^{[2]}}$$
> >  
> > $$\frac{\partial L}{\partial a^{[2]}} = a^{[2]} - y$$
> >
> > $$\frac{\partial a^{[2]}}{\partial z^{[2]}} = f'(z)$$
> >
> > $$\frac{\partial z^{[2]}}{\partial w^{[2]}} = a^{[1]}$$
> >  
> >  where $$w$$ are weights of layers, $$b$$ is the bias, $$z^{[l]} = w^{[l]}a^{[l-1]} + b^{[l]}$$ and $$a$$ is the activation after putting $$z$$ in a non linearity
> >
> > Like these, we can derive the contribution of any factor to the loss function by the chain rule, which is used by computers to nudge these values in order to minimize the loss function. But this is in the form of scalars here; won't it be overly complex in a real neural network with matrices and thousands of parameters? Well, not so much.
> >
> > ### For an actual network
> > Consider the network structure I made for this project. Now I will use capital letters for variables, as they are now matrices, but the chain rule is valid here too.
> >
> >  $$\frac{\partial L}{\partial Z^{[2]}} = A^{[2]} - Y$$
> >
> > $$\frac{\partial L}{\partial W^{[2]}} = \frac{1}{m} dZ^{[2]} \cdot (A^{[1]})^T$$
> >
> > It is essentially the same chain rule as the simplified case, but now that there are matrices, we have to use the transpose to match the shapes; the rest is all the same
> >
> > $$\frac{\partial L}{\partial b^{[2]}} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[2]}$$
> >
> > $$\frac{\partial L}{\partial Z^{[1]}} = (W^{[2]})^T \cdot dZ^{[2]} * f'(Z^{[1]})$$
> >
> > Okay. So this is the math; now let's jump into coding.
> >
> ### Coding Progress
> > Completed the code today, adding backpropagation entirely and a gradient descent function.
> > Also added a tracker, which tracks accuracy every 50 iterations
> >
> > Faced errors causing the accuracy to be 11% at the 1st as well as 500th iteration. **Fixed**: Caused by an error in the order of parameters given to the `update_parameters()` function. I now use the same sequence for parameters so as not to get confused.
> >
> > As I had some extra time, I added a feature where the tuned weights and biases are stored in the same folder in `.npz` format. I found from browsing that an `npz` file would be way faster for NumPy to read compared to a spreadsheet or `.csv` file
> >
# Some errors and problems faced by me
> 
| Problem  |  Fix  |
|----------|-------|
|Importing and processing the MNIST csv file.| Imported Samson Zhuang's method for the same, also understood why his method is effective|
|Understanding the need on onehot and how to implement it as a code| Significance explained by ChatGPT and imported code from geeksforgeeks|
|Shape error in backpropagation| Recursive debugging and improvising, some know-how taken from stackoverflow as well|
|While running the code, the accuracy stays at 11.18% after 0 iteration as well as after 500 iterations | Softmax function was not working properly, as $$e^{n}$$ was overflowing resulting in NaN, so the model was not improving. Corrected by substracting the max logit first as : `z = z - z.max(axis=1, keepdims=True)`
|Even after implementing the above change, the network was stuck at 11.18%| After much debugging, found that the function parameters of `gradient_descent()` were in incorrect order|
|The variables of my code (dW1, dW2) were not been able to import to PyTorch code | Reason is, name of my code has hyphen in, which python always treats as minus sign. To avoid this, make a new python file in your folder and copy paste the code. Now import the variables.|

# Epilogue
To be honest, this was a very interesting project. On Saturday, I felt overwhelmed by the math and complex algorithm, but eventually got to understand it's essence. This project helped me grasp fundamentals of aconcept that can be applied to such a diversity. To be frank, I chose this project, because I knew from the surface level, that it is an extremely powerful algorithm in machine learning, and now I have basis for that belief.

About task 1.4, I did not have enough time to learn and code in PyTorch, so I used ChatGPT codex to code that for me, which then I used to compare gradient matrix values of my code and the PyTorch code by importing dW1, dW2 from my code to PyTorch code. As mentioned above, there will be error if you directly import it, but following the fix of the problem, you will be able to see for yourself too.

<p align="center">Thank You</p>
