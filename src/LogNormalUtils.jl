function get_lognormal_params(desired_μ, desired_σ)
    μ²=desired_μ^2
    σ²=desired_σ^2
    ln_μ=log(μ²/sqrt(μ²+σ²))
    ln_σ=sqrt(log(1+(σ²/μ²)))
    return ln_μ, ln_σ
end

function get_lognormal_desc(d::LogNormal)
    return mean(d), std(d)
end