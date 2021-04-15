function conditional_bayes_err = compute_cond_bayes_err(theta, margins, risk_gibbs,u)

joint_bayes_risk = compute_joint_bayes_risk(margins, risk_gibbs, theta);
prob_be_labeled = sum(margins > theta)/u;
if prob_be_labeled == 0
    prob_be_labeled = 1e-15;
end
conditional_bayes_err = joint_bayes_risk / prob_be_labeled;
     
end