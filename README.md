# mlp-tf-node
Multi-Layer Perceptron implementation in Javascript (Node JS) using the Tensorflow library

# Introduction

This class is meant to be an abstraction of an Multi-Layer Perceptron inspired in the 
'Machine Learning An Algorithimic Perspective' by Stephen Marsland in Javascript (Node JS) 
using the Tensorflow library.

The book has its own implementation using Python and Numpy you can check it out here:
https://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html

In this moment the MLP class is not very costumizable since you can only
set:
- the hidden layer activation function (sigmoid is the default)
- the output layer activation function (linear is the default)
- the learning rate ()
- the training algorithm (sgd, momentum, adam)
- the loss function

# Examples:

## Function aproximation

``` Javascript
const p = new MLPerceptron([[0], [1], [2], [3], [4]], [[1, 2, 3, 4, 5]], 5, 
                           'sgd', 'sigmoid', 'linear', 'meanSquaredError',
                            0.25);

p.earlyStoppingTraining(2000, 0.000001, 0).then( (h) => {
  p.predict([[0]]);
  p.predict([[1]]);
  p.predict([[2]]);
  p.predict([[3]]);
  p.predict([[4]]);
});
```

*Result:*
```
>>> Training stopped  [ 0.008052811957895756 ] [ 0.007291194051504135 ] [ 0.007092813495546579 ] 83
>>> Prediction:  [ 0.9779470562934875 ]
>>> Prediction:  [ 1.9224510192871094 ]
>>> Prediction:  [ 2.9995110034942627 ]
>>> Prediction:  [ 4.014418601989746 ]
>>> Prediction:  [ 4.788935661315918 ]
``` 

## XOR

```Javascript
const p2 = new MLPerceptron([[0,0], [0,1], [1,0], [1,1]], [[0,1,1,0]], 5, 
                            'adam', 'sigmoid', 'sigmoid', 'meanSquaredError', 
                            0.25);
                            
p2.train(200, 0).then( (h) => {
  const conf = p2.confMatrix(p2.inputs, p2.targets);
  conf.print();
  console.log('Precision: ' + p2.precision * 100 + '%');
});
```

*Result:*
```
>>> epoch: 0
>>> loss: 0.25319623947143555
>>> epoch: 100
>>> loss: 0.0007259364938363433
>>> Training stopped  [ 0.0003494261181913316 ]
>>> Tensor
>>>     [[2, 0],
>>>     [0, 2]]
>>> Precision: 100%
```