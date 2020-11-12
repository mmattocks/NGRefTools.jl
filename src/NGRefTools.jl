module NGRefTools
    using ConjugatePriors, Distributions, Plots

    include("NGRef.jl")
    export marginals
    include("NIGRef.jl")
    include("MarginalTDist.jl")
    export MarginalTDist, MTDist_MC_func, plot_n_MTDist, plot_logn_MTDist, mean_mass_comparator, log_mean_mass_comparator
end # module
