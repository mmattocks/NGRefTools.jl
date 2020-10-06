module NGRefTools
    using Distributions

    export GenTDist,PMM_GenTDist

    struct GenTDist{F<:Real} <: ContinuousUnivariateDistribution
        t::TDist{F}
        μ::F
        σ::F

        function GenTDist{F}(ν::F,μ::F,σ::F) where {F<:Real}
            new(TDist(ν),μ,σ)
        end
    end

    dof(d::GenTDist) = d.t.ν
    params(d::GenTDist) = (d.t.ν,d.μ,d.σ)
    @inline partype(d::GenTDist{F}) where {F<:Real} = F

    mean(d) = d.μ; median(d) = d.μ; mode(d) = d.μ 

    function rand(gt::GenTDist)
        return gt.μ+rand(gt.t)*gt.σ
    end

    function quantile(gt::GenTDist, p)
        results=Vector{Float64}()
        q=quantile(gt.t,p)
        q.*=gt.σ
        return q.+=gt.μ
    end

    #posterior marginal of the mean for a normal gamma posterior from a uninformative reference prior
    function PMM_GenTDist(x::AbstractVector)
        n=length(x)
        μ=mean(x)
        ssr=sum((x.-μ).^2)
        σ=sqrt(ssr/(n(n-1)))
        GenTDist(n-1,μ,σ)
    end

end # module
