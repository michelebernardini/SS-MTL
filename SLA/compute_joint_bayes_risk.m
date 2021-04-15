function infimum = compute_joint_bayes_risk(margins, risk_gibbs, theta)

    sampling_rate = 50;
    dummi_sample_rate = 0:sampling_rate-1;

    u = numel(margins);
    gammas = theta + (1-theta)*(dummi_sample_rate+1)/sampling_rate;
    infimum = 1e+05;
    upper_bounds = [];
    
    for n = 1:numel(gammas)
       gamma = gammas(n);
       prob_between = sum((margins < gamma) & (margins > theta))/u;
       K = risk_gibbs + 0.5*(sum(margins)/u-1);
       %M-less of gamma
       Mg = sum(margins(margins < gamma))/u;
       %M-no-greater of theta
       Mt = sum(margins(margins <= theta))/u;
       A = K + Mt - Mg;
       upper_bound = prob_between + (A*(A > 0))/gamma;
       upper_bounds = [upper_bounds;upper_bound];
       if upper_bound < infimum
            infimum = upper_bound;
       end
       if n > 3
            if (upper_bounds(end) > upper_bounds(end-1)) && (upper_bounds(end-1) >= upper_bounds(end-2))
                break
            end
       end
       
    end

end