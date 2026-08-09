---
title: "Learning Backpropagation and implementing a neural network to replicate 1 bit binary XOR operation"
date: 2026-08-09
draft: false
---

https://www.youtube.com/watch?v=VMj-3S1tku0

^karpathy’s intro to neural networks & backpropagation

i wanted to understand ‘attention is all you need’ paper. here’s a resource - https://nlp.seas.harvard.edu/annotated-transformer/

the problem is that i couldn’t understand much of it. it kinda makes sense vaguely but it’s not clear why a specific choice was made. so, my task right now is to understand neural networks to an extent where i can implement it by my own.

here’s colab link for my replication of karpathy’s micrograd and then using it to train a neural network that can result in a binary XOR operation

https://colab.research.google.com/drive/1JvOFrZqrvruzUj1mK7cIsp0f2X9XCigW?usp=sharing

the goal of the video is to build something like pytorch but at a smaller scale, very small. pytorch uses ‘tensors’ as the main ‘objects’. all the operations are done on these tensors. a vector has a list of elements in one dimension. a matrix has a list of elements in 2 dimensions. a tensor is capable of having a list of elements in more than 2 dimensions. we can’t imagine a 4-d structure but if we can simply extrapolate whatever we see till 3d, just consider the possibility of being able to increase the dimensions to whatever number we want mathematically. and because we wanna build micrograd based on this, we will create our object as ‘Value’ - nothing complicated, it’s just a number. all the operations will be performed on this ‘Value’ object. actually, watching the video would be simpler than me explaining everything in this blog here. here’s a partial code snippet for how we define the ‘Value’ object.

```python

class Value:

  def __init__(self, data, _children=()):

    # initialize the 'Value' object with the data & children where children stores the info about previous objects which got together to create this object

    self.data = data

    self._prev = set(_children) # 'set' is used instead of 'tuple' because we do not care about duplicates & order; duplicates are handled while calculating gradient

    self.grad = 0 # gradient is initialized as 0 at first and will be updated as required

    self._backward = lambda: None # _backward() will be implemented individually for each operation

    # _op -> operations are ignored because i'm not gonna implement the graphviz part here

```

here’s the core idea:

let’s consider an equation `d = a*b + c`; if i increase ‘a’ with a very small value ‘h’, how will it influence the value of ‘d’? the expectation here is that we’re going to replace our question with an equation where we’ll find the coefficients of the equation that will somehow satisfy the question. as for the current blog, my goal is to calculate XOR of 2 inputs using a neural network implying that i want to figure out the coefficients of an equation that’ll give me the answer without even knowing the logic behind XOR.

anyways, as for the equation d = a*b + c, the answer is simple. we differentiate ‘d’ wrt ‘a’ to understand how a small change in the value of ‘a’ will influence the value of ‘d’. we know that `f’(x) = (f(x+h) - f(x))/h`

```python
a = 2

b = -3

c = 10

d = a*b + c


h = 0.00001

a+=h

d2 = a*b + c

print((d2-d)/h) # differentiating wrt 'a' will give you slope as 'b' anyways.

# the question here is that what will happen to 'd' if 'a' is increased slightly

# the sign here tells us whether the value increases/decreases

print(d2)
```

```python
-3.000000000064062
3.9999699999999994
```
because the coefficient of ‘a’ is negative, the result of (a*b+c) is expected to be slightly less than what it used to be. this information is also provided by the fact that derivative of ‘d’ wrt ‘a’ at that point is -3. the derivative tells us that by updating the value of ‘a’ in the direction of slope will decrease the overall value of ‘d’.

this is the reason behind including a gradient in the ‘Value’ object. the gradient is supposed to tell us how to update the value of a particular object so that the final value which is a result of many operations on the previous objects can become closer to the expected value.

https://www.youtube.com/watch?v=2-mzxsSWVCU&list=PL2zRqk16wsdo3VJmrusPU6xXHk37RuKzi

^another resource that teaches about neural networks.

NOTE: watch both these resources before reading this blog. 

let’s start with the smallest unit of a neural network/MLP(Multi Layer Perceptron) - neuron/perceptron.

![perceptron intro example](/3/perceptron_intro_example.png)

as shown in the above picture, let’s say we want to classify or divide the plane into 2 parts. consider this as a binary classification problem. x1 and x2 are 2 inputs that can take any value and then the function f(x1,x2) should allow us to tell whether (x1,x2) will be part of the right side or left side of that line. how can we mathematically figure out the coefficients of x1 and x2 that will help us with this problem? let’s call the coefficients as w1 and w2 respectively. this will make the function `f(x1,x2) = w1*x1 + w2*x2` but then if this is all we’ve got, every (0,0) will simply result in 0. but there is obviously a possibility where we might want the (0,0) to be present in a different place. so, let’s add a threshold kinda thing which will compensate for the lack of this possibility. let’s call this “bias” and denote it with ‘b’. now, the function f(x1,x2) becomes `w1*x1 + w2*x2 + b`.

these values - *(w1, w2 and b)* are called as **parameters** of a neural network. our job is to figure out appropriate values of these so called parameters which will help us with classification of the 2d space into 2 parts. let’s assume both the weights to be -2 and the bias to be 3. now the function becomes `-2x1-2x2+3`. 

but this isn’t the end of this function. because we want to classify the 2d space into 2 different parts, we’ll say that anything that results in a negative value or zero will be considered as part of left side and anything that produces positive value will be considered as part of the right side. at this point, there’s still a chance you’re wondering what even is the point of all this stuff.

![parameter count data of some popular models](/static/3/parameter_count_examples.png)

^[https://labelyourdata.com/articles/llm-fine-tuning/llm-model-size]



![red & blue binary classification graph](/static/3/image.png)

take a look at this above image. we’ve got a bunch of points and there are 2 kinds - red and blue. if i ask you to find a way to separate these 2 groups, the obvious choice is to simply draw a straight line between them. as long as you don’t find any contradictional data point to this, this simple straight line can act as a boundary that separates these two groups. it is worth acknowledging that this isn’t the only way to classify the points. you can move the line slightly leftwards or rightwards or maybe tilt a bit and still make sure that the red and blue points are on opposite sides. the truth is there is no one right answer. as long as we’re happy with it, we’ll call it a day. but here comes the main part - how do you make a computer be able to draw this line of separation?

we gotta represent everything in a mathematical format and the way we tried to do this binary classification for this particular example is by writing a function like

`f(x1,x2) = w1*x1 + w2*x2 + b`.

and then by through observation or maybe trial and error - let's say we came up with the coefficients for this equation to be

`f(x1,x2) = -2x1-2x2+3`

the way a computer is going to classify if a point belongs to the red group or the blue group is to calculate this value and then check if it’s greater than 0 or not. so, the final function(let’s call this an activation function) becomes :

if(-2x1-2x2+3)>0 -> it belongs to red group
or else it belongs to blue group

so, these parameters - the weights and biases - w1,w2 and b are the values that allow us to make a prediction using a perceptron or a neuron.

![sama about parameter size](/static//3/image%20copy.png)

when people talk about parameters, these weights and biases are what they mean

and, is this enough? nope. our classification problem can be difficult for a single perceptron to handle. look at the below classification.

![spiral classification graph sample](/static/3/image%20copy%202.png)

if you were tasked to classify these red and blue points, you’d draw a circle around the blue points. anything inside the circle belongs to the blue group while everything outside of this circle will be part of the red group. now, will a linear equation be able to classify this? no matter what line you draw here, a single line cannot classify the points.

![lines solving spiral classification](/static/3/image%20copy%203.png)

so, what if you could draw more lines using more perceptrons/neurons. any point that lies inside the triangle made by these 3 lines will be part of the blue group and rest of the points will belong to the red group. this tells you that if your model can use more neurons at once, then there’s a chance that you might be able to solve even difficult problems. you can keep increasing the number of neurons, call it a neural network and solve all your problems. all? maybe or maybe not.

![neural network generic image](/static/3/image%20copy%204.png)

[https://www.geeksforgeeks.org/deep-learning/neural-network-node/]


look at the arrangement of neurons in the above figure. you can arrange a bunch of neurons as a layer. you can have the input layer. and then the output of these neurons will pass as input to the next layer of neurons. and then as you keep doing this for layer after layer, you can have the final output layer. usually, you’ll be using something called an activation function at the last layer to allow us to make our predictions easier. remember the activation function i mentioned in the first example -

so, the final function(let’s call this an activation function) becomes :

if(-2x1-2x2+3)>0 -> it belongs to red group
or else it belongs to blue group

this activation function is basically a step function.

![step function](/static/3/image%20copy%205.png)

this step function does the job well for our current example as long as we try to predict if a data point belongs to a certain class in our classification problem. but it doesn’t give you much information when you try to move your data point slightly. i mean, when you move your data point to some extent, it stays 0 for a while but then suddenly, at some point, it’ll just start showing up 1. it’s impossible to understand how the small change in the data point is changing the final output. this is something we might be interested in - to be able to see how changing the input slightly influences the final output. the issue with this step function is that it changes itself abruptly. so, in order to avoid this problem, we tend to pick smoother activation functions - something like sigmoid or maybe softmax.

![sigmoid function](/static/3/image%20copy%206.png)

these smooth functions allow us to observe the continuous change in the output as we change the input. but then, how exactly does it “tell” us? *gradients*.

let’s say we’ve got some data to learn from. you initialize the parameters of the neural network with some random numbers and then make a random prediction and then compare it with the true result. when you compare, you can quantify how bad your prediction is, as compared to the true expectation. and then call this quantification of the comparison as ‘loss function’. for example, one of the simplest way to quantify difference between expectation and prediction is MSE - Mean Squared Error. you simply subtract both the values and if you care more about the absolute value of the error instead of the sign, you can choose to square it. find the average of all the ‘loss’ results over data sample. now that we’ve quantified the loss, we can think about decreasing the loss and try to make sure that our expectation gets closer to reality.

remember the d = a*b + c equation example at the start? differentiate the function wrt ‘a’. you get dd/da = b. if ‘b’ is negative, just as shown in the example where b=-3, the final value of ‘d’ decreases slightly when ‘a’ increases slightly.

so, if our goal was to decrease ‘d’ slightly, we’ll have to increase ‘a’ slightly and then re-measure the loss to see if our expectation matches reality. for every iteration, all we do is try to decrease the loss by adjusting the parameters appropriately. and so this is why our ‘Value’ object has ‘grad’, to tell us how a neuron is influencing the final loss function.


now that we’ve decided to measure gradients, we need to figure out the order in which we do it.

![backpropagation generic example](/static/3/image%20copy%207.png)

[https://www.youtube.com/watch?v=An5z8lR8asY]


to make a prediction, all we do is calculate the output of each neuron and pass it to the next neurons appropriately as an input. keep doing this till we get the final output, the loss function, which is supposed to be minimized. this process is called forward pass.

for each neuron, we move back to its inputs( the neurons which operated together which are then passed as input to this neuron) and calculate how the previous layer is influencing/affecting it. the neurons in the previous layer can be stored as ‘_prev’ in the ‘Value’ object. if you keep going backwards, you can calculate the gradients/influence the previous layer has on the current layer. but because our goal is to minimize the final loss function, we can use chainrule in differentiation to calculate the effect of any particular neuron on the final loss function.

![chain rule visualization](/static/3/image%20copy%208.png)

[https://en.wikipedia.org/wiki/Chain_rule]


let’s look at how we multiply two ‘Value’ objects.
```python
  def __mul__(self, other):

    other = other if isinstance(other, Value) else Value(other)

    out = Value(self.data * other.data, (self, other))


    def _backward():

      self.grad += other.data * out.grad

      other.grad += self.data * out.grad

    out._backward = _backward

    return out
```

the first line is to make sure that if you’re multiplying the ‘Value’ object with a numerical, you can turn that into a ‘Value’ object before making it interact with the existing ‘Value’ object. the second line is just multiplying the values of both the objects and storing it as a new ‘Value’ object. but right now, our task is to understand backpropagation - the _backward() function. remember the function d=a*b+c and how dd/da = b

the derivative of ‘d’ wrt ‘a’ is the coefficient of ‘a’, i mean, the ‘other object’ that is being multiplied by it. but what we wanted to know is how a particular neuron will influence the final output(loss function) and not just the immediately next neuron. so, by using the chain rule, we multiply this gradient with the gradient of the output gradient which has already taken the chainrule into consideration wrt the loss function.
one more important part is the accumulation of gradients. consider this equation b=2*a. this can also be written as b = a+a
if you directly calculate db/da you’ll get 2, but if for some reason, you treated each ‘a’ as a different object in this equation, sure you’ll still get db/da as 2 but notice how you add gradient of each object(the same object but represented as 2 different objects). if we had simply written
```python
self.grad = other.data * out.grad
```
we would have individually calculated the gradient of a ‘Value’ object and overwritten the information of whatever that object was influencing the final loss function. we do not want to overwrite the effect because even though the object might be represented differently in different places, the object is still the same object and the object’s previous gradient information has to be included by accumulating, wherever that object’s gradient is being calculated.


now that we know how the parameters(or the weights & biases) have to be updated in order to minimize the loss, we do this:
```python
parameter.data += -learning_rate*parameter.grad
```

the negative sign indicates that we’re trying to decrease the loss by changing the *value* in the *opposite direction* of the gradient. the learning rate denotes how big we wanna update the value of the neuron. if you change the parameter’s value by a small number, the final loss function will also be changed by a small number and we can see if we’re updating the parameters appropriately.

![learning rate image](/static/3/image%20copy%209.png)

^[https://www.jeremyjordan.me/nn-learning-rate/]


our task is to find the lowest point. if the learning rate is too high, we might keep skipping the lowest point. if our learning rate is too low, we might spend too much time to get to the lowest point. even though we can still find the lowest point, we care about getting to the lowest point quicker. so, the optimal choice here is to, maybe start with a suitably small learning rate and then as you feel you’re getting closer to the lowest point, you wanna make the learning rate even smaller so as to not skip anything accidentally.

it’s worth noting that there’s no right answer while working on some problems. you just have to keep making better decisions as you go along and that’s just how life is.

here’s an overview of what we want to do:

- create ‘Value’ object with numerical value, gradient
- make the object remember the previous objects which operated together to result in this object
- create functions so that the ‘Value’ object can operate with each other.
- create a _backward() function for each of these allowed operations on the ‘Value’ object in order to calculate gradients of each object
- create a backward() function that takes a ‘Value’ object and iteratively performs _backward() function on each previous ‘Value’ objects to help with backpropagation

```python
import math


class Value:

  def __init__(self, data, _children=()):

    # initialize the 'Value' object with the data & children where children stores the info about previous objects which got together to create this object

    self.data = data

    self._prev = set(_children) # 'set' is used instead of 'tuple' because we do not care about duplicates & order; duplicates are handled while calculating gradient

    self.grad = 0 # gradient is initialized as 0 at first and will be updated as required

    self._backward = lambda: None # _backward() will be implemented individually for each operation

    # _op -> operations are ignored because i'm not gonna implement the graphviz part here


  def __add__(self, other):

    other = other if isinstance(other, Value) else Value(other) # turning int/float into 'Value' object if needed

    out = Value(self.data + other.data, (self, other))


    def _backward():

      self.grad += out.grad # we accumulate gradients in case the same object is performing arithmetic in more than one ways (eg: b = a + a; each 'a' object has a gradient but we can accumulate it and consider the eqn as b = 2*a)

      other.grad += out.grad

    out._backward = _backward

    return out


  def __radd__(self, other): # note: 'r' denotes right operand and not 'reverse'

    return self + other


  def __neg__(self):

    return self*-1


  def __sub__(self, other):

    return self + (-other)


  def __rsub__(self, other):

    other + (-self)


  def __mul__(self, other):

    other = other if isinstance(other, Value) else Value(other)

    out = Value(self.data * other.data, (self, other))


    def _backward():

      self.grad += other.data * out.grad

      other.grad += self.data * out.grad

    out._backward = _backward

    return out


  def __rmul__(self, other):

    return self*other


  def __pow__(self, other):

    assert isinstance(other, (int, float)) # currently works for int/float only

    out = Value((self.data)**other, (self,))


    def _backward():

      self.grad += (other*(self.data**(other-1))) * out.grad

    out._backward = _backward

    return out


  def __truediv__(self, other):

    return self*(other**-1)


  def __rtruediv__(self, other):

    return other*(self**-1)


  def exp(self):

    x = self.data

    out = Value(math.exp(x), (self,))


    def _backward():

      self.grad += out.data * out.grad

    out._backward = _backward

    return out


  def tanh(self):

    x = self.data

    t = (math.exp(2*x)-1)/(math.exp(2*x)+1)

    out = Value(t, (self,))


    def _backward():

      self.grad += (1-t**2)*out.grad

    out._backward = _backward

    return out


  def relu(self):

    out = Value(0 if self.data<0 else self.data, (self, ))


    def _backward():

      self.grad += (out.data>0)*out.grad

    out._backward = _backward

    return out

 

  def sigmoid(self):

    out = Value(1/(1+self.exp(-self.data)))


    def _backward():

      self.grad += out.data * (1-out.data)

    out._backward = _backward

    return out


  def __repr__(self):

    return f"Value(data = {self.data}, grad = {self.grad})"


  def backward(self):

    topo = []

    visited = set()


    def build_topo(v):

      if v not in visited:

        visited.add(v)

        for child in v._prev: # for every visited node, go through all its children(previous elements) and find if they have children(prev elements, kinda like the parents actually: the ones that 'performed' operation and resulted in this object) too

          build_topo(child)

        topo.append(v) # once there's no more children(previous elements/objects), add it to the topological sorted list


    build_topo(self) # start building the topological sorted list with this object

    self.grad = 1 # initialize the current object's grad as 1 ; how much does this value change when 'final value' is incremented with infinitesimally small value; because this object is same as the final object, it's just going to be 1


    for element in reversed(topo): # reversing the list because backpropagation goes 'backward'; you can't calculate how the current object might influence the output when output hasn't even been calculated yet; so, finding out how the previous object might influence the already calculated output is a sensible thing to do here;

      element._backward()


```

- create a neuron that can take certain number of inputs and result in an output
- create a layer of these neurons
- create an MLP(Multi Layer Perceptron). i mean, combine those layers of neurons together.(try to make it a bit similar to pytorch)

```python
import random


class Module:

  def zero_grad(self):

    for p in self.parameters():

      p.grad = 0 # resetting all the grads to 0 before every iteration to avoid grads getting accumulated after every forward pass


  def parameters(self):

    return []


class Neuron(Module): # each neuron has certain number of inputs and an intrinsic bias

  def __init__(self, nin, nonlin=True):

    self.w = [Value(random.uniform(-1,1)) for i in range(nin)] # creating a list of nin(number of inputs) Value objects with random weights initialized between -1 and 1

    self.b = Value(0) # initializing with 0 bias; why not random here?

    self.nonlin = nonlin


  def __call__(self, x): # forward pass

    act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b) # calling a neuron gives us the output of neuron => sum(wi*xi) + bias;

    return act.relu() if self.nonlin else act # peform relu operation as an activation if nonlinearity is preferred


  def parameters(self):

    return self.w + [self.b] # returning the list of all weights with the bias appended at the end


  def __repr__(self):

    return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module): # each layer has certain number of neurons which will get inputs from the previous layer of neurons; nin is the number of neurons in previous layer that this layer gets its inputs from and nout is the number of neurons in this layer

  def __init__(self, nin, nout, nonlin = True):

    self.neurons = [Neuron(nin, nonlin) for i in range(nout)]


  def __call__(self, x):

    out = [neuron(x) for neuron in self.neurons]

    return out


  def parameters(self): # a list of parameters of all the neurons appended

    param = []

    for neuron in self.neurons:

      for parameter in neuron.parameters():

        param.append(parameter)

    return param


  def __repr__(self):

    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

  def __init__(self, nin, nouts): # nin is the number of inputs just like in the layer, nouts is the list of sizes of each layer, i mean, basically nouts is a list of ['nout' per layer]

    sz = [nin] + nouts # just appending the number of inputs at the start

    self.layers = [Layer(sz[i], sz[i+1], nonlin = (i!=len(nouts)-1)) for i in range(len(nouts))] # activation function isn't applied at the last layer


  def __call__(self, x):

    for layer in self.layers:

      x = layer(x) # output of current layer will be treated as input of next layer

    return x


  # def parameters(self):

  #   param = []

  #   for layer in self.layers:

  #     for neuron in layer.neurons:

  #       for parameter in neuron.parameters():

  #         param.append(parameter)

  #   return param

  def parameters(self):

    param = []

    for layer in self.layers:

      param += layer.parameters()

    return param


  def __repr__(self):

    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
```

let us try to use a single neuron and see if it can figure out the parameters in order to produce a XOR binary output.

```python
# task: run backpropagation on a single neuron 100 times and note down the loss for every 10 runs


# the neuron gets 2 inputs called i1 and i2 (naming odd to avoid conflicting with previous cases)

# i1 = Value(0)

# i2 = Value(1) ; i don't think we even need these i1 & i2 when we're gonna have the size of each input as 2 anyways and the neuron is initialized with 2 inputs


# our goal is to figure out values of w1, w2 and b (weights & biases) which will produce XOR result by doing w1*i1+w2*i2+b

# we expect it to fail in this case of using linear function & later will try out using a non linear activation function


def loss_MSE(a, b):

  return (a-b)**2


neuron1 = Neuron(2, nonlin = False) # creating a neuron with 2 possible inputs or (2 nodes serving as an input to this neuron)

x_input1 = [[0,0], [0,1], [1,0], [1,1]] # defining the input states

y_true1 = [0, 1, 1, 0] # defining output states corresponding to the input states

for epoch in range(1,101):

  loss = Value(0) # initializing loss with 0

  learning_rate = 0.01 # fractional change

  for x,y in zip(x_input1, y_true1):

    inputs = [Value(v) for v in x]

    current_output = neuron1(inputs) # calling neuron1 with inputs to run forward pass

    expected_output = Value(y)

    loss += loss_MSE(expected_output, current_output) # using Mean Squared Error function to calculate loss; (will use BCE later)

  neuron1.zero_grad() # resetting gradients of the neuron to 0 before every backpropagation

  loss.backward() # running backpropagation on the loss function

  for parameter in neuron1.parameters():

    parameter.data += -learning_rate*parameter.grad # updating parameters according to the gradients calculated during backpropagation

  if epoch%10 == 0:

    print("<<>>")

    print(loss.data)

    print(neuron1.parameters())


```


```python
output:

<<>>

1.5484803366390487

[Value(data = 0.7044742467461127, grad = 1.0735201596466282), Value(data = -0.06649658889077949, grad = -0.49988970900009067), Value(data = 0.09605630930023884, grad = -0.7137974740770598)]

<<>>

1.3899639904797816

[Value(data = 0.6016699582294542, grad = 0.9769285295911079), Value(data = -0.028269346472818962, grad = -0.30866188816863316), Value(data = 0.15338987069885676, grad = -0.49189991491924534)]

<<>>

1.2802012210468465

[Value(data = 0.5114517843554243, grad = 0.8405003298028826), Value(data = -0.003254491506451692, grad = -0.20992064134380306), Value(data = 0.1969294086672929, grad = -0.3984264910080473)]

<<>>

1.202578867604001

[Value(data = 0.43458210154181015, grad = 0.7123295219401165), Value(data = 0.0140296000007111, grad = -0.145940889368249), Value(data = 0.23351351360194958, grad = -0.3421625431646125)]

<<>>

1.1472994104278949

[Value(data = 0.3695887051076002, grad = 0.601580533182775), Value(data = 0.025966692229834998, grad = -0.0996888808534806), Value(data = 0.26529829089953727, grad = -0.2992569753921619)]

<<>>

1.10770287969357

[Value(data = 0.3147107069011617, grad = 0.5080052031191962), Value(data = 0.033946504330777304, grad = -0.0649829653917926), Value(data = 0.29319373080392586, grad = -0.2631526294910226)]

<<>>

1.0791770019060387

[Value(data = 0.26834717815226433, grad = 0.42939706242868825), Value(data = 0.0389423830845365, grad = -0.03877598872994015), Value(data = 0.3177485099344905, grad = -0.23177047024882869)]

<<>>

1.058508819427605

[Value(data = 0.22913145281241734, grad = 0.36341893612291165), Value(data = 0.0416910329929665, grad = -0.01911253289637571), Value(data = 0.3393813849895677, grad = -0.2042246964487624)]

<<>>

1.0434491217704445

[Value(data = 0.1959167983318926, grad = 0.30800532219658305), Value(data = 0.0427643283858337, grad = -0.004550738917822983), Value(data = 0.35844483611518224, grad = -0.17997676182226785)]

<<>>

1.0324153522236497

[Value(data = 0.16774527021510557, grad = 0.26141178834394196), Value(data = 0.042608551714518536, grad = 0.006030730179478638), Value(data = 0.37524526923634677, grad = -0.1586139757062961)]
```

the loss seems to start at 1.5 approximately and then stops decreasing much when it reaches around 1. this implies that when the neuron came up with randomized parameters, the loss was high and we updated the parameters appropriately which decreased the loss for a while but then the neuron showed no real signs of improvement after a certain point. the loss plateaued. we expect the loss to be closer to 0. that’s not happening here. let’s try to run 1000 iterations and only print the loss this time for every 100 iterations.

```python
1.0154520895390924

1.0011739685634249

1.000092748482278

1.0000073933198959

1.0000005905191118

1.0000000471865729

1.0000000037708978

1.0000000003013565

1.0000000000240834

1.0000000000019247 

```

after 100 iterations, it basically shows no improvements even if you keep going for a 1000 iterations. this failure is expected behaviour.

![XOR 1 bit graph](/static/3/image%20copy%2010.png)


```python
[0, 0]: 0

[0, 1]: 1

[1, 0]: 1

[1, 1]: 0 

```

take a look at the expected classification. using 2 neurons would allow us to classify these 2 different kinds of outputs for this XOR operation with inputs as 0 and 1. let’s directly create a multi layer perceptron `nn = MLP(2,[2,1])`

this says that our neural network takes 2 inputs. we have 1 hidden layer with 2 neurons and these will produce a single output in the end. here below is the network representation.

![MLP(2,[2,1]) visual representation](/static/3/image%20copy%2011.png)

we started with the goal to replicate 1 bit binary XOR operations using a neural network. when i first ran those 100 iterations with a sigmoid activation function, the output was still not good enough. it seemed like the loss was getting stuck after a while but the truth is that it wasn’t. the loss was getting smaller but the 100 iterations and slow learning rate wasn’t enough.


i changed the learning rate from 0.001 to 0.05 and then i increased the number of epochs to 2000. here’s the loss for every 100 iterations.

```python
def loss_MSE(a, b):
  return (a-b)**2

nn2 = MLP(2, [2,1])
x_input3 = [[0,0], [0,1], [1,0], [1,1]] # defining the input states
y_true3 = [0, 1, 1, 0] # defining output states corresponding to the input states
for epoch in range(1,2001):
  loss = Value(0) # initializing loss with 0
  learning_rate = 0.05 # fractional change
  for x,y in zip(x_input3, y_true3):
    inputs = [Value(v) for v in x]
    current_output = nn2(inputs) # calling neuron1 with inputs to run forward pass
    expected_output = Value(y)

    loss += loss_MSE(expected_output, current_output[0]) # using Mean Squared Error function to calculate loss; (will use BCE later)
  nn2.zero_grad() # resetting gradients of the neuron to 0 before every backpropagation
  loss.backward() # running backpropagation on the loss function
  for parameter in nn2.parameters():
    parameter.data += -learning_rate*parameter.grad # updating parameters according to the gradients calculated during backpropagation
  if epoch%100 == 0:
    # print("<<>>")
    print(loss.data)
    # print(nn.parameters())

```

output:

```python
1.0026311846275422

1.001244343661032

1.0002009063965311

0.9991881224959122

0.9979617023493701

0.9962108723753565

0.9934004962004473

0.9884668991831536

0.979190661969258

0.9611370984931396

0.9267616481666417

0.8662571880929302

0.7699431366821801

0.628253563460216

0.43111658093269944

0.20854376363383176

0.060689850502583916

0.011312234107011154

0.0016442145087271682

0.0002149624146973919 
```

as you can see, the loss is very close to 0 - seems good enough for us. let us run a forward pass and see the output of neural network.

```python
x_input3 = [[0,0], [0,1], [1,0], [1,1]] # defining the input states

y_true3 = [0, 1, 1, 0] # defining output states corresponding to the input states

for x,y in zip(x_input3, y_true3):

    inputs = [Value(v) for v in x]

    current_output = nn2(inputs)

    print(current_output)
```

```python
[Value(data = 0.004976356806394744, grad = 0)]

[Value(data = 0.9931211466651677, grad = 0)]

[Value(data = 0.9931309604867424, grad = 0)]

[Value(data = 0.009555650308542774, grad = 0)] 
```

**NOTE**: when i pasted the code at the top of this blog, there are a few mistakes i made - failing to accumulate gradients or not returning the output properly or improperly implementing _backward(), etc. and i just wanna leave it like that. the final(current) colab code is the one i used by the end of this blog which gave me the right output.

