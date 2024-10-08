using Documenter
using DistributionsAD

makedocs(;
    sitename="DistributionsAD",
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    doctest=false,
)

deploydocs(; repo="github.com/TuringLang/DistributionsAD.jl.git", push_preview=true)
