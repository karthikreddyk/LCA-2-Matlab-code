%function R_updated = agentlearning(A,B,Q,R,R_est,u_obs,X,theta_max,theta_min, atAgent, ForAgent)
function R_updated = agentlearningforMIMO(A,B,Q,R,R_est,u_obs,X,theta_max,theta_min,ForAgent)
% for now considering only one input at each agent
% this function provides an update for learnt theta's for each agent
% this is the crux of learning algorithm
% atAgent - where is learning taking place
% ForAgent - for whom learning is taking place
% estimation for u1 is done at u1 as well, it doesn't matter where the
% learning is being carried out.
% running the learning loop 4 times per one iteration of control loop
% u_obs - observed value of control by (atAgent) of (ForAgent)
Control_R = R_est; 
Control_R_MAX = R_est; 
Control_R_MIN = R_est;
num_of_ips = length(u_obs);

if (ForAgent == 1)
    relevant_indices = 1:num_of_ips;
else if (ForAgent == 2)
        relevant_indices = num_of_ips+1:2*num_of_ips;
    end
end


Control_R_MAX(relevant_indices,relevant_indices) = theta_max*eye(num_of_ips);
Control_R_MIN(relevant_indices,relevant_indices) = theta_min*eye(num_of_ips);

%Control_R(atAgent,atAgent) = R (atAgent,atAgent); % since estimation is being done at 'atAgent'


% finding exterems and intial slope

[Sinf_thetaMax,L_thetaMax,G_thetaMax] = dare(A,B,Q,Control_R_MAX);
[Sinf_thetaMin,L_thetaMin,G_thetaMin] = dare(A,B,Q,Control_R_MIN);
[Sinf_thetaEst,L_thetaEst,G_thetaEst] = dare(A,B,Q,Control_R);
U_max = - G_thetaMax * X; u_max = norm(U_max(relevant_indices));
U_min = - G_thetaMin * X; u_min = norm(U_min(relevant_indices));
U_est = - G_thetaEst * X; u_est = norm(U_est(relevant_indices));
u_obs = norm(u_obs);
mulfac = 1; % multiplication factor, takes values 1 or -1 dep on some rules
theta_est = R_est(relevant_indices(1),relevant_indices(1));

if (abs(u_est - u_obs) < 1e-8)
    R_updated = R_est;
    return;
end
    

slope = (u_max - u_min)/(theta_max - theta_min);
endpoint1 = [theta_est; u_est];


if (slope > 0) && (u_obs < u_est)
    mulfac = -1;
end

if (slope < 0) && (u_obs > u_est)
    mulfac = -1;    
end

if (mulfac == 1)
    endpoint2 = [theta_max;u_max];
end

if (mulfac == -1)
    endpoint2 = [theta_min;u_min];
end
    
for i=1:5 % learning rate
    Invslope =  (endpoint2(1) - endpoint1(1))/(endpoint2(2) - endpoint1(2));
%    theta_est_new = theta_est +  Invslope * (endpoint2(2) - endpoint1(2));
    theta_est_new = endpoint1(1) +  Invslope * (u_obs - endpoint1(2));
    Control_R(relevant_indices,relevant_indices) = theta_est_new*eye(num_of_ips);
   [Sinf_thetaEst,L_thetaEst,G_thetaEst] = dare(A,B,Q,Control_R);
    U_est_new = - G_thetaEst * X;
    u_est_new = norm(U_est_new(relevant_indices));
    
%    if ((abs(u_est_new) - abs(u_obs)) * (abs(u_obs) - abs(endpoint2(2)))) > 0
    if (((u_est_new) - (u_obs)) * ((u_obs) - (endpoint2(2)))) > 0
        endpoint1 = [theta_est_new; u_est_new];
    else
        endpoint2 = [theta_est_new; u_est_new];
    end
    theta_est = theta_est_new;
%    Control_R(ForAgent,ForAgent) = theta_est; 
end
R_updated = Control_R;
end
%
 
