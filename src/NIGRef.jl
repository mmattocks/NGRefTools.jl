"""
    fit(NormalInverseGamma, x)

Return the posterior `NormalInverseGamma` distribution on the unknown mean and variance of a Normal model of data vector `x`, assuming an uninformative (reference) prior.

Example:

    julia> fit(NormalInverseGamma,rand(10))
    NormalInverseGamma{Float64}(mu=0.5052725306750604, nu=10.0, shape=4.5, rate=0.5057485400844524)

Reference: Kevin P. Murphy, Conjugate Bayesian Analysis of the Gaussian Distribution. 2007. https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
"""
    function Distributions.fit(::Type{NormalInverseGamma},x::AbstractVector)
        n=length(x)
        Vn = 1/n
        μ=mean(x)
        an = -.5 + (n/2)
        bn = .5 * (sum(x.^2)-(μ^2/Vn))
        return NormalInverseGamma(μ,Vn,an,bn)
    end

    function Distributions.params(d::NormalInverseGamma)
        return d.mu, d.v0, d.shape, d.scale
    end

    function marginals(d::NormalInverseGamma)
        m_T=MarginalTDist(d.mu,sqrt((d.shape*inv(d.v0))/d.scale),TDist(2*d.shape))
        m_Ga=InverseGamma(d.shape,d.scale)
        return m_T,m_Ga
    end

    # function NGplot(d::NormalInverseGamma; grid=1000, upper=.95, lower=.05)
    #     marg_mean, marg_var = marginals(d)
    #     xmax=quantile(marg_mean,upper)
    #     xmin=quantile(marg_mean,lower)
    #     ymax=quantile(marg_var,upper)
    #     ymin=quantile(marg_var,lower)
    #     xs=LinRange(xmin, xmax, grid)
    #     ys=LinRange(ymin, ymax, grid)
    #     zs=[pdf(d,x,y) for x in xs, y in ys]

    #     return surface(xs,ys,zs)
    # end
