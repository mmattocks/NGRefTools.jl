module NGRefTools
    using ConjugatePriors, Distributions, Plots
    #import Makie:surface

    include("NGRef.jl")
    export marginals
    include("NIGRef.jl")
    #export NGplot
    include("MarginalTDist.jl")
    export MarginalTDist, MTDist_MC_func, plot_n_MTDist, plot_logn_MTDist, mean_mass_comparator, log_mean_mass_comparator
    include("LogNormalUtils.jl")
    export get_lognormal_params, get_lognormal_desc
end # module
