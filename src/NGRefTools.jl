module NGRefTools
    using ConjugatePriors, Distributions

    include("NGRef.jl")
    include("NIGRef.jl")
    include("MarginalTDist.jl")
    export MarginalTDist, MTDist_MC_func
end # module
