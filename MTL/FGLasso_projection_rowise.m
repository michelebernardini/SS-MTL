%% FUNCTION FGLasso_projection_rowise
%   projection of sparse fused group Lasso.
%
%% OBJECTIVE
%   argmin_w { 0.5 \|w - v\|_2^2
%          + lambda_1 * \|w\|_1 + lambda_2 * \|w*R\|_1  +
%          + lambda_3 * \|w\|_2 }


function w = FGLasso_projection_rowise(v, lambda_1, lambda_2, lambda_3)

% starting point (dual variable).
w0 = zeros(length(v)-1, 1);

%% 1st Projection:
w_1 = flsa(v, w0,  lambda_1, lambda_2, length(v), 1000, 1e-9, 1, 6);

%% 2nd Projection:
nm = norm(w_1, 2);
if nm == 0
    w_2 = zeros(size(w_1));
else
    w_2 = max(nm - lambda_3, 0)/nm * w_1;
end

w = w_2;

end