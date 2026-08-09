---
title: "Learning Backpropagation and implementing a neural network to replicate 1 bit binary XOR operation"
date: 2026-08-09
draft: true
---

https://www.youtube.com/watch?v=VMj-3S1tku0

^karpathy’s intro to neural networks & backpropagation


i wanted to understand ‘attention is all you need’ paper. here’s a resource - https://nlp.seas.harvard.edu/annotated-transformer/


the problem is that i couldn’t understand much of it. it kinda makes sense vaguely but it’s not clear why a specific choice was made. so, my task right now is to understand neural networks to an extent where i can implement it by my own.


here’s colab link for my replication of karpathy’s micrograd and then using it to train a neural network that can result in a binary XOR operation

https://colab.research.google.com/drive/1JvOFrZqrvruzUj1mK7cIsp0f2X9XCigW?usp=sharing


the goal of the video is to build something like pytorch but at a smaller scale, very small. pytorch uses ‘tensors’ as the main ‘objects’. all the operations are done on these tensors. a vector has a list of elements in one dimension. a matrix has a list of elements in 2 dimensions. a tensor is capable of having a list of elements in more than 2 dimensions. we can’t imagine a 4-d structure but if we can simply extrapolate whatever we see till 3d, just consider the possibility of being able to increase the dimensions to whatever number we want mathematically. and because we wanna build micrograd based on this, we will call our object as ‘Value’ - nothing complicated, it’s just a number. all the operations will be performed on this ‘Value’ object. actually, watching the video would be simpler than me explaining everything in this blog here. here’s a partial code snippet for how we define the ‘Value’ object.
