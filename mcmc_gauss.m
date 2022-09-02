function [x_keep, L_keep, count, reflectivity, attenuation] = mcmc_gauss(A,R,fc,x0,xstep,xbnds,sigmaA,sigmaR,rho,Niter)
%
% [x_keep, L_keep, count, reflectivity, attenuation] = mcmc_gauss(A,R,fc,x0,xstep,xbnds,sigmaA,sigmaR,rho,Niter)
%
% subroutine for MCMC sampling using Metropolis-Hasting w/ normal
% distribution.
%
% Inputs:
% A = observed total attenuation (scalar - negative dB)
% R = observed reflectivity (scalar - dB)
% fc = radar center frequency (scalar - Hz)
% x0 = initial estimate of parameter vector (vector)
% xstep = step size in all parameter directions (vector)
% xbnds = bounds (array of upper and lower bounds)
% sigmaA = standard deviation of observed attenuation (attenuation
% uncertainty) (scalar - dB)
% sigmaR = standard deviation of observed reflectivity (reflectivity
% uncertainty) (scalar - dB)
% rho = cross-correlation between reflectivity and attenuation
% Niter = number of iterations
%
% Outputs:
% x_keep = array of accepted model parameters
% L_keep = likelihood of model parameters
% count = number of accepted configurations. Acceptance ratio is count/Niter
% reflectivity = vector of modeled reflectivities (can compare to observed
% for sanity check that model is converging)
% attenuation = vector of modeled attenuation (can compare to observed
% for sanity check that model is converging)

    % Initialize storage variables
    count = 0;
    L_keep = zeros(Niter,1);
    x_keep = zeros(Niter+1,length(x0));
    reflectivity = zeros(1,Niter);
    attenuation = zeros(1,Niter);
    
    % Model constants
    c = 299792458;            % speed of light in a vaccum
    n_ice = 1.78;             % refractive index of solid ice
    omega = 2*pi*fc;          % radar angular center frequency
    lambda = c/(n_ice*fc);    % radar wavelength in ice
    e0 = 8.85418782e-12;      % permittivity of free space
    mu0 = 1.26e-6;            % magnetic permeability of vacuum
    sigma_ice = 3.5e-5;       % average conductivity of ice, Siemens per meter (estimated from NGT firn core measurements)
    
    % Calculate complex dielectric constants of water and ice
    % Option A assumes an interface between solid ice and saturated pore
    % space like you might expect for a water core inside an ice blob
    % Option B assumes an interface between unsaturated and saturated firn
    % that would be more appropriate for an aquifer and uses the Kovacs
    % relationship to calculate real permittivity from density derived from
    % modeled porosity
    % Option A 
    e_ice = 3.17 + 1i*(sigma_ice/(omega*e0));
    % Option B 
    % e_ice = (1+0.845*(0.917*(1-x0(2))))^2 + 1i*(sigma_ice/(omega*e0));
    e_water = 80 + 1i*(x0(1)/(omega*e0));
    
    % Calculate complex dielectric constant of firn water mix using the
    % CRIM/Exponential Model/Licktenecker-Rother Equation
    % Shape factor alpha set to 0.4 based on results in Sihvola, et al
    % (1985) "Mixing Formulae and Experimental Results for the Dielectric
    % Constant of Snow", but anything between 1/3 and 1/2 is justifiable
    % Also easy enough to sub in other mixing models here as desired
    alpha = 0.4;
    e_mix = ((1-x0(2))*e_ice^alpha + x0(2)*e_water^alpha)^(1/alpha);
    
    % Calculate the rough interface scattering loss based on Peters, et al
    % (2005) formula
    g = (4*pi*x0(4))/lambda;
    loss = 10*log10(exp(-g^2)*besseli(0,0.5*g^2));
    
    % Reflectivity in dB of a specular ice-saturated firn interface
    reflectivity(1) = 20*log10(abs((sqrt(e_ice)-sqrt(e_mix))/(sqrt(e_ice)+sqrt(e_mix))))+loss;
    % Total attenuation through saturated layer using the full-form
    % attenuation coefficient from Ulaby (2014) (see table 2-1)
    atten_coeff = omega*sqrt(0.5*mu0*e0*real(e_mix)*(sqrt(1+(imag(e_mix)/real(e_mix))^2)-1));
    attenuation(1) = 10*log10(exp(-2*atten_coeff*2*x0(3)));
    
    % Save initial model guesses
    x_keep(1,:) = x0;
    
    for k = 1:Niter
        
        % Print updates on inversion progress
        if k == Niter
            fprintf("%d\n", (k/Niter)*100);
        elseif mod((k/Niter)*100,10) == 0
            fprintf("%d...", (k/Niter)*100);
        end
        
        % Propose a new random set of model parameters based on step size
        xprop = mvnrnd(x_keep(k,:),diag(xstep.^2),1);
        flag = 0;
        for m = 1:length(xprop)
            if xprop(m) <= xbnds(1,m) || xprop(m) >= xbnds(2,m)
                flag = 1;
            end
        end
        
        if flag == 0  % if proposed model is within the bounds,
           
            % Get current reflectivity
            dcurr_R = reflectivity(k);
            dcurr_A = attenuation(k);
            
            % Calculate reflectivity and attenution for new proposed model
            % parameters
            e_water = 80 + 1i*(xprop(1)/(omega*e0));
            e_mix = ((1-xprop(2))*e_ice^alpha + xprop(2)*e_water^alpha)^(1/alpha);
            g = (4*pi*xprop(4))/lambda;
            loss = 10*log10(exp(-g^2)*besseli(0,0.5*g^2));
            dprop_R = 20*log10(abs((sqrt(e_ice)-sqrt(e_mix))/(sqrt(e_ice)+sqrt(e_mix))))+loss;
            atten_coeff = omega*sqrt(0.5*mu0*e0*real(e_mix)*(sqrt(1+(imag(e_mix)/real(e_mix))^2)-1));
            dprop_A = 10*log10(exp(-2*atten_coeff*2*xprop(3)));
            
            % calculate likelyhood ratio of proposed model
            p_d_xprop = (-1/(2*(1-rho^2)))*(((A-dprop_A)/sigmaA).^2 - ...
                         2*rho*((A-dprop_A)/sigmaA)*((R-dprop_R)/sigmaR)...
                         +((R-dprop_R)/sigmaR).^2);
            p_d_x = (-1/(2*(1-rho^2)))*(((A-dcurr_A)/sigmaA).^2 - ...
                         2*rho*((A-dcurr_A)/sigmaA)*((R-dcurr_R)/sigmaR)...
                         +((R-dcurr_R)/sigmaR).^2);
            ratio = p_d_xprop - p_d_x;
            % if proposed model is more likely than previous model, accept
            % it
            if ratio > 0
                count = count + 1;
                L_keep(k,:) = p_d_xprop;
                x_keep(k+1,:) = xprop;
                reflectivity(k+1) = dprop_R;
                attenuation(k+1) = dprop_A;
            else
                % Otherwise, set a random cutoff and keep or reject model
                % based on that
                u = log(rand(1,1));
                if ratio > u
                    count = count + 1;
                    L_keep(k,:) = p_d_xprop;
                    x_keep(k+1,:) = xprop;
                    reflectivity(k+1) = dprop_R;
                    attenuation(k+1) = dprop_A;
                else
                    x_keep(k+1,:) = x_keep(k,:);
                    L_keep(k,:) = p_d_x;
                    reflectivity(k+1) = reflectivity(k);
                    attenuation(k+1) = attenuation(k);
                end
            end
        else  
            % if proposed model is outside the bounds, keep the previous
            % model
            x_keep(k+1,:) = x_keep(k,:);
            reflectivity(k+1) = reflectivity(k);
            attenuation(k+1) = attenuation(k);
            dcurr_R = reflectivity(k);
            dcurr_A = attenuation(k);
            L_keep(k,:) = (-1/(2*(1-rho^2)))*(((A-dcurr_A)/sigmaA).^2 - ...
                         2*rho*((A-dcurr_A)/sigmaA)*((R-dcurr_R)/sigmaR)...
                         +((R-dcurr_R)/sigmaR).^2);
        end
    end

    fprintf("Acceptance Ratio: %f\n", count/Niter);
end