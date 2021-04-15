function theta_star = optimal_threshold(x_u, margins, risk_gibbs)

    sampling_rate = 50;
    u = size(x_u{1},1);
    
    %A set of possible thetas
    theta_min = min(margins);
    theta_max = max(margins);
    dummi_sample_rate = 0:sampling_rate-1;
    thetas = theta_min + dummi_sample_rate*(theta_max-theta_min)/sampling_rate;
    dummi_thetas = 0:numel(thetas)-1;

    for i = 1:numel(dummi_thetas)
        bayes_err(i) = compute_cond_bayes_err(thetas(i), margins, risk_gibbs,u);     
    end
    
    [~,min_idx] = min(bayes_err);
    theta_star = thetas(min_idx);

end
