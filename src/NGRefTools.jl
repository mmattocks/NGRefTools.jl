module NGRefTools
    using ConjugatePriors, Distributions

    include("MarginalTDist.jl")
    export MarginalTDist, fit, MTDist_MC_func

"""
    fit(NormalGamma, x)

Return the posterior `NormalGamma` distribution on the unknown mean and variance of a Normal model of data vector `x`, assuming an uninformative (reference) prior.

Example:

    julia> fit(NormalGamma,rand(10))
    NormalGamma{Float64}(mu=0.5052725306750604, nu=10.0, shape=4.5, rate=0.5057485400844524)

Reference: Kevin P. Murphy, Conjugate Bayesian Analysis of the Gaussian Distribution. 2007. https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
"""
    function Distributions.fit(::Type{NormalGamma},x::AbstractVector)
        n=length(x)
        μ=mean(x)
        ssr=sum((x.-μ).^2)
        α=(n-1)/2
        β=ssr/2
        return NormalGamma(μ,n,α,β)
    end

end # module
