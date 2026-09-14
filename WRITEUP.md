# Postman-Club-Recruitment
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
>> This version will be simple as I am new to all this and also due due to time constraint I am facing. It will contain only 1 hidden layer and would have this structure:
>>
>> Input Layer -----> Hidden layer (64 neurons) ----> Output layer
>>
>> This would result in a not-so-very efficient network with success rate of about 0.8 to 0.85 according to Gemini's prediction
>
> 
>### Problems faced
> > Could not implement onehot function by myself. Fixed.
> > Code becomes very complex and messy very quickly, difficulty in keeping track of all variables and function parameters. Still not sure about them all, as network is yet to test
> > Forgot backprop formulas. Revisited the handwritten notes I made yesterday. Fixed.
> > Still not very comfortable with shapes, have to think about them while writing every operation. **NOT FIXED**


## Day 3 - Monday, September 14

> ### Maths behind backprop
> Before starting with today's progress, let me present you the math behind the curtains
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
> > Like these, we can derive contribution of any factor to the loss function by chain rule, which is used by computer to nudge these values in order to minimize loss function. But this is in form of scalars here, won't it be overly complex in real neural network with matrices and thousands of parameters? Well, not so much.
> >
> > ### For an actual network
> > Consider the structure network I made for this project. Now I will use capital letters for variables as, now they are matrices, but still, chain rule is valid here too.
> >
> >  $$\frac{\partial L}{\partial Z^{[2]}} = A^{[2]} - Y$$
> >
> > $$\frac{\partial L}{\partial W^{[2]}} = \frac{1}{m} dZ^{[2]} \cdot (A^{[1]})^T$$
> >
> > It is essentially the same chain rule as the simplified case, but now that there are matrices, we have to use trnaspose to match the shapes, rest all same
> >
> > $$\frac{\partial L}{\partial b^{[2]}} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[2]}$$
> >
> > $$\frac{\partial L}{\partial Z^{[1]}} = (W^{[2]})^T \cdot dZ^{[2]} * f'(Z^{[1]})$$
> >
> > Okay. so this is the math, now let's jump into coding.
> >
> ### Coding Progress
> > Completed the code today, adding backpropagation entirely and gradient descent function.
> > Also added as tracker, which tracks accuracy every 50 iterations
> >
> > Faced errors causing the accuracy to be 11% at 1st as well as 500th iteration. **Fixed**: Caused due to error in order of parameters given to `update_parameters()` functions. I now used the same sequence for parameters to not get confused.
> >
> > 
