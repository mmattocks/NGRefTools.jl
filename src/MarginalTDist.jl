"""
    MarginalTDist(ν,μ,σ)

Return a shifted, scaled T Distribution with degrees of freedom `ν`, mean `μ`, and standard deviation `σ`.

This is the marginal distribution of the posterior mean for an uninformative (reference) Normal Gamma prior distribution. It is also the distribution of the posterior predictive for m=1 new observations.

    MarginalTDist(ν)        #Standard TDist with ν dof
    MarginalTDist(ν,μ,σ)    #TDist with ν dof shifted by μ, scaled by σ

    params(d)               #Return d.ν, d.μ, d.σ
    mean(d)                 #Return d.μ (median, mode are identical)
    std(d)                  #Return d.σ
    quantile(d,p)           #Return quantile at probability p

Reference: Kevin P. Murphy, Conjugate Bayesian Analysis of the Gaussian Distribution. 2007. https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
"""
struct MarginalTDist{F<:Real} <: ContinuousUnivariateDistribution
    t::TDist{F}
    μ::F
    σ::F

    MarginalTDist{F}(ν::F,μ::F,σ::F) where {F<:Real} = σ<=0 ? throw(DomainError("MarginalTDist σ must be >0!")) : new{F}(TDist(ν),μ,σ)
end

MarginalTDist(ν::Integer, μ::AbstractFloat, σ::AbstractFloat)=MarginalTDist{Float64}(float(ν), μ, σ)
MarginalTDist(ν::Real)=MarginalTDist{Float64}(float(ν),0.,1.)

Distributions.dof(d::MarginalTDist) = d.t.ν
Distributions.params(d::MarginalTDist) = (d.t.ν,d.μ,d.σ)
@inline Distributions.partype(d::MarginalTDist{F}) where {F<:Real} = F

Distributions.mean(d::MarginalTDist) = d.μ; Distributions.median(d::MarginalTDist) = d.μ; Distributions.mode(d::MarginalTDist) = d.μ
Distributions.std(d::MarginalTDist) = d.σ

function Distributions.rand(mt::MarginalTDist)
    return mt.μ+rand(mt.t)*mt.σ
end

function Distributions.quantile(mt::MarginalTDist, p)
    results=Vector{Float64}()
    q=quantile(mt.t,p)
    q.*=mt.σ
    return q.+=mt.μ
end

"""
    fit(MarginalTDist, x; PP=false)

Assuming an uninformative (reference) Normal Gamma prior, return the marginal posterior distribution of the mean of a Normal model of observations `x`. 

Equivalent to the frequentist sampling distribution of the MLE mean. If `PP=true`, return the posterior predictive distribution for m=1 additional observations sampled from the marginal posterior mean instead.

Example:

    julia> PMM_MarginalTDist(rand(10))
    MarginalTDist{Float64}(
    t: TDist{Float64}(ν=9.0)
    μ: 0.4789484996068401
    σ: 0.08491142725619677
    )

Reference: Kevin P. Murphy, Conjugate Bayesian Analysis of the Gaussian Distribution. 2007. https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
"""
function fit(::Type{MarginalTDist}, x::AbstractVector{<:Real}; PP=false)
    PP ? PPM_MTDist(x) : PMM_MTDist(x)
end

#Posterior marginal mean distribution
function PMM_MTDist(x)
    n=length(x)
    μ=mean(x)
    ssr=sum((x.-μ).^2)
    σ=sqrt(ssr/(n*(n-1)))
    return MarginalTDist(n-1,μ,σ)
end

#Posterior predictive mean distribution
function PPM_MTDist(x)
    n=length(x)
    μ=mean(x)
    ssr=sum((x.-μ).^2)
    αn=(n-1)/2
    βn=.5*ssr
    return MarginalTDist(2*αn,μ,(βn*(n+1))/(αn*n))
end

"""
    MTDist_MC_func(func, xs...; lower=.025, upper=.975, mc_its=1e6, summary=false)

Monte Carlo execute `func` with a random sample from a MarginalTDist fitted to each x in xs, over `mc_its` iterates, returning the calculated results (if `summary` is `true`) or the mean and quantiles specified by `lower` and `upper`.

Example:

    julia> NGRefTools.MTDist_MC_func(*,[rand(10),rand(10)],summary=true)
    (0.09491370397229557, 0.20712818357976473, 0.3385677491994645)

See also: [`fit(MarginalTDist,x)`](@ref)
"""
function MTDist_MC_func(func::Function, xs; lower=.025, upper=.975, mc_its=1e6, summary=false)
    dists=[fit(MarginalTDist,x) for x in xs]
    results=Vector{Float64}()
    for it in 1:mc_its
        push!(results, func(rand.(dists)...))
    end
    summary ? (return quantile(results, lower), mean(results), quantile(results,upper)) : (return results)
end
