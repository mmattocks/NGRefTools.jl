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
MarginalTDist=LocationScale{Float64,TDist{Float64}}

Distributions.dof(d::MarginalTDist) = d.ρ.ν

Distributions.std(d::MarginalTDist) = d.σ

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
function Distributions.fit(::Type{MarginalTDist}, x::AbstractVector{<:Real}; PP=false)
    PP ? PPM_MTDist(x) : PMM_MTDist(x)
end

#Posterior marginal mean distribution
function PMM_MTDist(x)
    n=length(x)
    μ=mean(x)
    ssr=sum((x.-μ).^2)
    σ=sqrt(ssr/(n*(n-1)))
    return MarginalTDist(μ,σ,TDist(n-1))
end

#Posterior predictive mean distribution
function PPM_MTDist(x)
    n=length(x)
    μ=mean(x)
    ssr=sum((x.-μ).^2)
    αn=(n-1)/2
    βn=.5*ssr
    return MarginalTDist(μ,sqrt((βn*(n+1))/(αn*n)),TDist(2*αn))
end

"""
    MTDist_MC_func(func, xs...; lower=.025, upper=.975, mc_its=1e6, summary=false)

Monte Carlo execute `func` with a random sample from a MarginalTDist fitted to each x in xs, over `mc_its` iterates, returning the calculated results (if `summary` is `true`) or the mean and quantiles specified by `lower` and `upper`.

Example:

    julia> NGRefTools.MTDist_MC_func(*,[rand(10),rand(10)],summary=true)
    (0.09491370397229557, 0.20712818357976473, 0.3385677491994645)

See also: [`fit(MarginalTDist,x)`](@ref)
"""
function MTDist_MC_func(func::Function, xs; lower=.025, upper=.975, mc_its=1e7, summary=false)
    dists=[fit(MarginalTDist,x) for x in xs]
    results=Vector{Float64}()
    for it in 1:mc_its
        push!(results, func(rand.(dists)...))
    end
    summary ? (return quantile(results, lower), mean(results), quantile(results,upper)) : (return results)
end

function plot_logn_MTDist(xs, colors, markers, labels, xlabel, ylabel; args...)
    plt=plot(;args...)
    for (x,color) in zip(xs,colors)
        min=floor(minimum(x)-.15*mean(x))
        max=ceil(maximum(x)+.20*mean(x))
        X=collect(min:max)
        pmdist=fit(MarginalTDist, log.(x))
        y=pdf(pmdist,log.(X))
        plot!(plt,X,y,ribbon=(y,zeros(length(y))),label=:none,color=color,xlabel=xlabel,ylabel=ylabel)
    end
    for (x,color,marker,label) in zip(xs,colors, markers,labels)
        pmdist=fit(MarginalTDist, log.(x))
        min=floor(minimum(x)-.15*mean(x))
        max=ceil(maximum(x)+.20*mean(x))
        X=collect(min:max)
        y=pdf(pmdist,log.(X))
        scaty=[mean(y) for x in 1:length(x)]
        plt=scatter!(plt,x,scaty,color=color, marker=marker, label=label)
    end

    return plt
end

function plot_n_MTDist(xs, colors, markers, labels, xlabel, ylabel; args...)
    plt=plot(;args...)
    for (x,color) in zip(xs,colors)
        pmdist=fit(MarginalTDist, x)
        min=quantile(pmdist, .01)
        max=quantile(pmdist, .99)
        X=collect(min:(max-min)/1000:max)
        y=pdf(pmdist,X)
        plot!(plt,X,y,ribbon=(y,zeros(length(y))),color=color,label=:none,xlabel=xlabel,ylabel=ylabel)
    end
    for (x,color,marker,label) in zip(xs,colors, markers,labels)
        pmdist=fit(MarginalTDist, x)
        scaty=[pdf(pmdist,quantile(pmdist,.25)) for x in 1:length(x)]
        plt=scatter!(plt,x,scaty,color=color, marker=marker,label=label)
    end

    return plt
end

function plot_mt_timeseries(X, obs, colors, markers, labels; log=false, plotargs...)
    plt=plot(;plotargs...)
    for (ovec, color, marker, label) in zip(obs, colors, markers, labels)
        log ? (mts=[fit(MarginalTDist,log.(ovec[t])) for t in 1:length(X)]) :
            (mts=[fit(MarginalTDist,ovec[t]) for t in 1:length(X)])
        if log
            means=[exp(mean(mt)) for mt in mts]
            lower=[exp(mean(mt))-exp(quantile(mt,.025)) for mt in mts]
            upper=[exp(quantile(mt,.975))-exp(mean(mt)) for mt in mts]
        else
            means=[mean(mt) for mt in mts]
            lower=[mean(mt)-quantile(mt,.025) for mt in mts]
            upper=[quantile(mt,.975)-mean(mt) for mt in mts]
        end

        scatter!(plt, vcat([[X[t] for i in 1:length(ovec[t])] for t in 1:length(X)]...),vcat(ovec...), marker=marker, color=color, markersize=2, label=nothing)
        plot!(plt, X, means, ribbon=(lower, upper), color=color, label=label)
    end
    return plt
end

function mean_mass_comparator(x,y;labels=["x","y"])

    xdist=fit(MarginalTDist,x)
    ydist=fit(MarginalTDist,y)
    if mean(xdist)>mean(ydist)
        frac=ccdf(xdist,mean(ydist))
        println("$(round(frac*100,digits=1))% of the marginal posterior mean density of $(labels[1]) is above the mean of $(labels[2]).")
    else
        frac=cdf(xdist,mean(ydist))
        println("$(round(frac*100,digits=1))% of the marginal posterior mean density of $(labels[1]) is below the mean of $(labels[2]).")
    end
end

function log_mean_mass_comparator(x,y;labels=["x","y"])
    xdist=fit(MarginalTDist,log.(filter(n->!iszero(n),x)))
    ydist=fit(MarginalTDist,log.(filter(n->!iszero(n),y)))
    if mean(xdist)>mean(ydist)
        frac=ccdf(xdist,mean(ydist))
        println("$(round(frac*100,digits=1))% of the marginal posterior mean density of $(labels[1]) is above the mean of $(labels[2]).")
    else
        frac=cdf(xdist,mean(ydist))
        println("$(round(frac*100,digits=1))% of the marginal posterior mean density of $(labels[1]) is below the mean of $(labels[2]).")
    end
end