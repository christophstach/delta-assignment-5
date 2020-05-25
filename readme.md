# 1 MLP Step by Step from Scratch
To add momentum we add the following lines of code to the **NNBPL_Complete_Cycle.mlx**:

```matlab
for i = 1:3
	dw(i + 6) = Eta * Delta(3) * o(i + 3);
	dw(i + 6) = dw(i + 6) + gamma * dw(i + 6); % Momentum
	w(i + 6) = w(i + 6) + dw(i + 6); %New
end
```

and:

```matlab
for i = 1:3
    dw(i) = Eta * Delta(1) * o(i);
    dw(i) = dw(i) + gamma * dw(i);
    w(i) = w(i) + dw(i);
end

for i = 4:6
    dw(i) = Eta * Delta(2) * o(i - 3);
    dw(i) = dw(i) + gamma * dw(i);
    w(i) = w(i) + dw(i);
end    
```

The following table shows the number of epochs needed to reach the desired error of 0.01 with different momentum term ($\gamma$) configurations: 

| Momentum term ($\gamma$) | Number of Epochs needed |
| ------------------------ | ----------------------- |
| 0                        | 2363                    |
| 0.8                      | 1320                    |
| 0.85                     | 1285                    |
| 0.9                      | 1251                    |
| 0.95                     | 1220                    |

As we can see, using $\gamma$ greatly increases the performance of the neural network training algorithm. Using a momentum term it needs less epochs to finish the training.

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


