module NGRefTools
    using ConjugatePriors, Distributions

    include("NGRef.jl")
    export marginals
    include("NIGRef.jl")
    include("MarginalTDist.jl")
    export MarginalTDist, MTDist_MC_func
end # module
