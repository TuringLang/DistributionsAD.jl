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
