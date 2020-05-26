# 1 MLP Step by Step from Scratch
To add momentum we add the following lines of code to the **NNBPL_Complete_Cycle.mlx**:

```matlab
for i = 1:3
	%                                          Momentum	
	dw(i + 6) = Eta * Delta(3) * o(i + 3) + gamma * dw(i + 6);
	w(i + 6) = w(i + 6) + dw(i + 6);
end
```

and:

```matlab
for i = 1:3
	%                                  Momentum
	dw(i) = Eta * Delta(1) * o(i) + gamma * dw(i)
	w(i) = w(i) + dw(i);
end

for i = 4:6
	%                                      Momentum
	dw(i) = Eta * Delta(2) * o(i - 3) + gamma * dw(i);
	w(i) = w(i) + dw(i);
end    
```

We also add a second hidden layer to the neural network. See code in (NNBPL_Complete_Cycle.mlx)[https://github.com/christophstach/delta-assignment-5/blob/master/NNBPL_Complete_Cycle.mlx].

Afterwards we tested the performance of the neural network  with different configurations. We test different $\gamma$ values with the original neural network provided and the version which has an additional hidden layer. The following table shows the results of the experiments:

| Momentum term ($\gamma$) | Hidden Layers | Number of Epochs needed |
| ------------------------ | ------------- | ----------------------- |
| 0                        | 1             | 2363                    |
| 0.9                      | 1             | 242                     |
| 0.95                     | 1             | 133                     |
| 0                        | 2             | 13077                   |
| 0.90                     | 2             | 1411                    |
| 0.95                     | 2             | 987                     |

As we can see, using $\gamma$ greatly increases the performance of the neural network training algorithm with both configurations of hidden layer. Using momentum helps to find the desired maximum error in less number of epochs. Adding an additional layer doesn't help to increase the performance of the network. The reason for that is that a deeper neural network is harder to calculate and optimize than a shallow neural network. The gradients in the earlier layer vanish faster. In our use case the problem can be easily solved by a shallow neural network. Other use cases thought could be more complex, so that the shallow neural network is not able to model the problem. In this case adding more layer is beneficial to get a higher accuracy even if we sacrifice training and inference speed.

## 2 Function Approximation

I implemented the $f(x) = x * sin(x)$ as a lambda function.  

```matlab
f = @(x) x .* sin(x);
```

With the lambda I was able to generate the samples need to train the neural network and also to generate the true values, which are needed to compare them to the predicted values. I calculated the Root mean Squared Error in the following way:

```matlab
rmse = @(y_pred, y_true) sqrt(mean((y_pred - y_true).^2));
```

The table bellow shows different experiments with different configurations. In general it can be said that using more training samples is beneficial for a lower RMSE.

| Number of Neurons | Number of Samples | RMSE | Note                                                         |
| ----------------- | ----------------- | ---- | ------------------------------------------------------------ |
| 3                 | 6                 | 1.40 | Underfitting due to few number of samples. Overall good performance an generalization ability. |
| 5                 | 6                 | 1.27 | Not enough samples. Starts to overfit.                       |
| 20                | 6                 | 5.67 | Strong evidence of overfitting.                              |
| 5                 | 3                 | 4.41 | Not enough samples.                                          |
| 5                 | 10                | 0.40 | Good generalization ability.                                 |
| 5                 | 20                | 0.02 | Almost perfect generalization ability (Best model, lowest RMSE). |



The diagrams for different number of neurons:

<img src="plots\n3s6.png" style="zoom:48%;" />
<img src="plots\n5s6.png" style="zoom:48%;" />
<img src="plots\n20s6.png" style="zoom:48%;" />

The diagrams for different number of samples:

<img src="plots\n5s3.png" style="zoom:48%;" />
<img src="plots\n5s10.png" style="zoom:48%;" />
<img src="plots\n5s20.png" style="zoom:48%;" />




# 3 Environment / Situation Classifier

No results yet




# Results

 * Task 1: I was able to implement the momentum. I also was able to implement the forward path for an additional layer in the neural network. I was struggling implementing the $\delta$ Error-Signal for the earlier layer. So I couldn't calculate the backward pass need for Subtask 2, 3 and 4.
 * Task 2: I was able to completely do the assignment. The plots with graphs needed are in the plot folder.
 * Task 3: I didn't understand how to train the Classifier without any data. I seems like we should generate the needed data our self. But had no idea how to do that for the given situation


