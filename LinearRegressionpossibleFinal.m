%---------------------CPU Performance: Linear Regression ---------------------------------
importfile('data2.mat');

n = 5;  %SPECIFY # OF FEATURES FOR CALCULATING GD OF ALL FEATURES SIMULTANEOUSLY
X = data2(:, 1:n+1);  %X = CPU DATA OF ONLY INTEGER-BASED FEATURES 
Y = data2(:, n+1);    % Y = "Published Relative Performance" (Benchmark Scores)
    
% Do you want feature normalization?
normalization = true;


% Applying mean normalization to our dataset
if (normalization)
    for i = 1:n
        x(:, i) = (X(:, i) - max(X(:, i))) / (max(X(:, i)) - min(X(:, i)));
    end
    y = Y;
    y = (Y - max(Y)) / (max(Y) - min(Y));
else
    x = X;
    y = Y;
end   %RESULTS IN COLUMNS OF x THAT HAVE NORMALIZED VALUES OF DATASET


% Adding a column of ones to the beginning of the 'x' matrix
%------------------ONLY RUN ONE TIME----------------------------
x = [ones(length(x), 1) x];
%------------------ONLY RUN ONE TIME----------------------------



% Plotting the dataset
%figure;
%plot(X(:, 2), Y, 'rx', 'MarkerSize', 10);
%xlabel('Size ( squared feet )');
%ylabel('Benchmark Score');
%title('CPU Hardware'); 
 
 
% Running gradient descent on the data
% 'x' is our input matrix
% 'y' is our output matrix
% 'parameters' is a matrix containing our initial theta and slope
 
parameters = zeros(n+1, 1);
%best learning rate = 0.065
learningRate = .15;   %too small -> slow convergence. too large -> fail to converge
repetition = 150;
[parameters, costHistory] = gradient(x, y, parameters, learningRate, repetition);

% Plotting our cost function on a different figure to see how we did
figure;
plot(costHistory, 1:repetition);

% Plotting our final hypothesis:

%parameters(1) is the intercept of our hypothesis line or Theta_1
%parameters(2) is the slope of our hypothesis line or Theta_2
figure;
plot(min(X(:, 3)):max(X(:, 2)), parameters(1) + parameters(2) * (min(X(:, 2)):max(X(:, 2))));
xlabel('Size ( squared feet )');
ylabel('Benchmark Score');
title('CPU HARDWARE'); 
hold on;
 


%Plotting the dataset on the same figure
plot(X(:, 2), Y, 'rx', 'MarkerSize', 10);
 
% Plotting our cost function on a different figure to see how we did
figure;
plot(costHistory, 1:repetition);
 
% Finally predicting the output of the provided inputs

%Add a one to the beginning of the desired inputs to test
%so that matrix sizes match up (we added a column of 1 to our x earlier)
%inputs = [1; 300; 1000; 16000; 8; 2; 112];
%inputs = [1; 112; 1000; 1000; 1; 4];    %should be 8
inputs = [1; 25; 2000; 12000; 3; 5];    %should be 66
%inputs = [1; 23; 32000; 64000; 32; 64];    %should be 1100
%inputs = [1; 75; 2000; 16000; 1; 38];    %should be ....
%inputs = [1; 220; 1000; 8000; 1; 2];    %should be 70something
%inputs = [1; 17; 4000; 16000; 6; 12];    %should be 133

if (normalization)    %normalize the inputs
    for i = 1:n
        inputs(i) = (inputs(i) - max(X(:, i))) / (max(X(:, i)) - min(X(:, i)));
    end
end

%inputs = [105; 2000; 8000; 16; 4; 14; 1];
disp(inputs);
disp(parameters);
%param = parameters(2:n, 1);
%output = parameters(1) + param' * inputs;
output = parameters' * inputs;
disp(output);
