using RBMmagic
using Documenter

DocMeta.setdocmeta!(RBMmagic, :DocTestSetup, :(using RBMmagic); recursive=true)

makedocs(;
    modules=[RBMmagic],
    authors="Yiming Lu <luyimingboy@163.com> and contributors",
    sitename="RBMmagic.jl",
    format=Documenter.HTML(;
        canonical="https://Rose_max111.github.io/RBMmagic.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Rose_max111/RBMmagic.jl",
    devbranch="main",
)
