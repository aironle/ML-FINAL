% Plotting the datasets
importfile('data2.mat');
n = 5;  %SPECIFY # OF FEATURES FOR CALCULATING GD OF ALL FEATURES SIMULTANEOUSLY
X = data2(:, 1:n);  %X = CPU DATA OF ONLY INTEGER-BASED FEATURES 
Y = data2(:, n+1);    % Y = "Published Relative Performance" (Benchmark Scores)


% Plotting MYCT vs Benchmark Score
plot(X(:, 1), Y, 'rx', 'MarkerSize', 10);
xlabel('MYCT: machine cycle time in nanoseconds');
ylabel('Benchmark Score');
title('CPU Hardware'); 

% Plotting MMIN vs Benchmark Score
plot(X(:, 2), Y, 'rx', 'MarkerSize', 10);
xlabel('MMIN: minimum main memory in kilobytes');
ylabel('Benchmark Score');
title('CPU Hardware'); 

% Plotting MMAX vs Benchmark Score
plot(X(:, 3), Y, 'rx', 'MarkerSize', 10);
xlabel('MMAX: maximum main memory in kilobytes');
ylabel('Benchmark Score');
title('CPU Hardware'); 

% Plotting CACH vs Benchmark Score
figure;
plot(X(:, 4), Y, 'rx', 'MarkerSize', 10);
xlabel('CACH: cache memory in kilobytes');
ylabel('Benchmark Score');
title('CPU Hardware'); 

% Plotting CHMIN vs Benchmark Score
figure;
plot(X(:, 5), Y, 'rx', 'MarkerSize', 10);
xlabel('CHMIN: minimum channels in units');
ylabel('Benchmark Score');
title('CPU Hardware'); 

% Plotting CHMAX vs Benchmark Score
figure;
plot(X(:, 6), Y, 'rx', 'MarkerSize', 10);
xlabel('CHMAX: maximum channels in units');
ylabel('Benchmark Score');
title('CPU Hardware'); 
