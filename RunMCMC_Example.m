clear;
clc;

%%%%%%%%%%%%%% Set Up and Run MCMC Inversion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = -3.1642;       % Observed averaged attenuation in dB
R = -18.8431;      % Observed average reflectivity in dB
sigmaA = 0.1071;   % Uncertainty in total attenuation in dB
sigmaR = 3.8263;   % Uncertainty in reflectivity in dB
rho = 0.824;       % Cross-correlation between reflectivity and attenuation
fc = 300e6;        % Radar center frequency, Hz
% Note: I calculated rho value by using as the empirical correlation
% between A and R for all 19 ice blob water tables - this would need to be
% updated if you're using a different data set

% Model parameters bounds (min and max values that would be reasonable -
% these should be updated based on the specific physical conditions on your
% site)

% Conductivity estimates based on Kendrick et al (2018), Doyle et al
% (2018), and Bartholomew et al (2011)
min_conductivity = 3e-4;   % Conductivity of stored water, S/m
max_conductivity = 8e-3;   
min_porosity = 0.1;    % Porosity of dry firn before saturating
max_porosity = 1;
min_thickness = 0.6;   % Thickness of water saturated layer in meters
max_thickness = 20;    
min_roughness = 0;     % Roughness of ice-water interface in meters
max_roughness = 0.5;

% Guesses to initialize the model (doesn't really matter what these are as
% long as they fall in the bounds set above - inversion should converge
% given long enough regardless of starting point)
init_conductivity = 5e-3;
init_porosity = 0.2;
init_thickness = 5;
init_roughness = 0;

% Number of iterations to run the inversion (usually want at least 1 million)
Niter = 5000000;  
% Initial gussed model parameters
x0 = [init_conductivity init_porosity init_thickness init_roughness];  
% Allowable ranges for model parameters
xbnds = [[min_conductivity min_porosity min_thickness min_roughness]; ...
    [max_conductivity max_porosity max_thickness max_roughness]]; 
% Step size for walking through model parameters space (may need to tune
% these to speed up or improve convergence)
xstep = [1e-4 0.01 0.1 0.01];

% Run MCMC code
[x_keep, L_keep, count, reflectivity, attenuation] = mcmc_gauss(A,R,fc,x0,xstep,xbnds,sigmaA,sigmaR,rho,Niter);

%%%%%%%%%%%%%%%%%%%%%% Visualize Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot log-likelyhood for each iteration
figure;
plot(L_keep);
title("Log Likelihood of Accepted Model by Iteration");
ylabel("Log Likelyhood");
xlabel("Iteration Number");
set(gca,'FontSize',20);

% Note: for marginal distributions you may want to drop the first 100-200
% iterations as the model "burns in" and starts to converge to reasonable
% values - can be estimated from looking at convergence of log likelihood
% plot

% Plot the marginal distributions
figure;
histogram(x_keep(:,1)*((1e6)/100));
title("Marginal Distribution of Conductivity");
xlabel('Conductivity (\muS/cm)');
ylabel('PDF');

figure;
histogram(x_keep(:,2));
title("Marginal Distribution of Porosity");
xlabel('Porosity');
ylabel('PDF');

figure;
histogram(x_keep(:,3));
title("Marginal Distribution of Thickness");
xlabel('Saturated Layer Thickness (m)');
ylabel('PDF');

figure;
histogram(x_keep(:,4));
title("Marginal Distribution of Roughness");
xlabel('Water-Ice Interface Roughness (m)');
ylabel('PDF');

figure;
histogram(reflectivity, 'Normalization', 'pdf');
hold on;
line([R R], [0 1], 'Color', 'r', 'LineStyle', '--');
hold on;
line([R+sigmaR R+sigmaR], [0 1], 'Color', 'r', 'LineStyle', '--');
hold on;
line([R-sigmaR R-sigmaR], [0 1], 'Color', 'r', 'LineStyle', '--');
title('Modeled Reflectivity vs. Observed Reflectivity');
legend('Histogram of modeled reflectivity', 'Observed Mean', 'Observed \sigma');

figure;
histogram(attenuation, 'Normalization', 'pdf');
hold on;
line([A A], [0 5], 'Color', 'r');
hold on;
line([A+sigmaA A+sigmaA ], [0 5], 'Color', 'r', 'LineStyle', '--');
hold on;
line([A-sigmaA A-sigmaA ], [0 5], 'Color', 'r', 'LineStyle', '--');
title('Modeled Attenuation vs. Observed Attenuation');
legend('Histogram of modeled attenuation', 'Observed Mean', 'Observed \sigma');


