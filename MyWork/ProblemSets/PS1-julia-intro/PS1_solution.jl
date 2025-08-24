# ECON 6343 – PS1 solution (q1–q5)
# AI note (required by syllabus): e.g., “Used ChatGPT to draft and debug q2/q4 tests.”
# Run:  julia --project=. PS1_solution.jl

using Random, LinearAlgebra, Statistics
using CSV, DataFrames
using FreqTables
using Distributions
using JLD
using Test
# --- name helpers ---
hascol(df::AbstractDataFrame, s::Symbol) = s in names(df)

# Case-insensitive, punctuation-insensitive name resolver
normalize_key(s::AbstractString) = replace(lowercase(s), r"[^a-z0-9]+" => "_")
function find_col(df::AbstractDataFrame, target::AbstractString)
    key = normalize_key(target)
    table = Dict(normalize_key(String(n)) => n for n in names(df))
    get(table, key, nothing)  # returns the actual Symbol in df (or nothing)
end


# ---------- q1 ----------
function q1()
    Random.seed!(1234)
    A = -5 .+ 15 .* rand(10,7)           # U[-5,10]
    B = -2 .+ 15 .* randn(10,7)          # N(-2,15)
    C = hcat(A[1:5,1:5], B[1:5,6:7])     # 5x7
    D = map(x -> x <= 0 ? x : 0.0, A)

    println("nelem(A) = ", length(A))
    println("nunique(D) = ", length(unique(vec(D))))

    E = vec(B)
    F = Array{Float64,3}(undef,10,7,2); F[:,:,1]=A; F[:,:,2]=B
    F = permutedims(F, (3,1,2))          # 2 x 10 x 7
    G = kron(B, C)                       # C ⊗ F undefined (F is 3D) → skip

    JLD.save("matrixpractice.jld","A",A,"B",B,"C",C,"D",D,"E",E,"F",F,"G",G)
    JLD.save("firstmatrix.jld","A",A,"B",B,"C",C,"D",D)
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))
    CSV.write("Dmatrix.dat", DataFrame(D, :auto); delim='\t')
    save_tsv("Dmatrix.dat", DataFrame(D, :auto))
    return A,B,C,D
end

# ---------- q2 ----------
function q2(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix)
    @assert size(A)==(10,7) && size(B)==(10,7)
    @assert size(C)==(5,7)

    # (a) AB via loop, AB2 vectorized
    AB = similar(A)
    for i in axes(A,1), j in axes(A,2)
        AB[i,j] = A[i,j]*B[i,j]
    end
    AB2 = A .* B

    # (b) Cprime (loop) and Cprime2 (vectorized)
    Cprime = Float64[]
    for x in C
        if -5 <= x <= 5
            push!(Cprime, x)
        end
    end
    Cprime2 = vec(C[(C .>= -5) .& (C .<= 5)])

    # (c) Build X (N x K x T)
    N,K,T = 15169,6,5
    X = Array{Float64,3}(undef,N,K,T)
    X[:,1,:] .= 1.0
    X[:,5,:] .= rand(Binomial(20,0.6), N)
    X[:,6,:] .= rand(Binomial(20,0.5), N)
    for t in 1:T
        p = 0.75 * (6 - t) / 5
        X[:,2,t] = rand(N) .< p
        μ3, σ3 = 15 + t - 1, 5*(t-1)
        X[:,3,t] = μ3 .+ (σ3==0 ? zeros(N) : σ3 .* randn(N))
        μ4, σ4 = pi*(6 - t)/3, 1/exp(1)
        X[:,4,t] = μ4 .+ σ4 .* randn(N)
    end

    # (d) β (K x T) via comprehension
    β = [ k==1 ? 1 + 0.25*(t-1) :
          k==2 ? log(t) :
          k==3 ? -sqrt(t) :
          k==4 ? exp(t)-exp(t+1) :
          k==5 ? t : t/3
          for k in 1:K, t in 1:T ]

    # (e) Y via comprehension
    Y = [ sum(X[n, :, t] .* β[:, t]) + 0.36*randn()
          for n in 1:N, t in 1:T ]

    # cache for tests / inspection
    global _Q2_cache = (; AB,AB2,Cprime, Cprime2, X, β, Y)
    return nothing
end

# ---------- q3 ----------

function q3()
    infile = "nlsw88.csv"
    @assert isfile(infile) "Place nlsw88.csv next to this script."

    nlsw = CSV.read(infile, DataFrame; missingstring=[".", "NA", "", "nan", "NaN"])

    # resolve columns robustly (case/spacing-insensitive)
    nm  = find_col(nlsw, "never_married")
    mar = find_col(nlsw, "married")
    col = find_col(nlsw, "collgrad")
    grd = find_col(nlsw, "grade")
    rac = find_col(nlsw, "race")
    ind = find_col(nlsw, "industry")
    occ = find_col(nlsw, "occupation")
    wag = find_col(nlsw, "wage")

        # --- (b) Percent never married & percent college grads ---
    pct_never = if hascol(nlsw, :never_married)
        mean(skipmissing(nlsw.never_married) .== 1)
    elseif hascol(nlsw, :married)
        mean(skipmissing(nlsw.married) .== 0)
    else
        @warn "Could not find never_married or married in: $(names(nlsw))"
        missing
    end

    pct_coll = if hascol(nlsw, :collgrad)
        mean(skipmissing(nlsw.collgrad) .== 1)
    elseif hascol(nlsw, :grade)
        # treat grade >= 16 as “college grad”
        mean(skipmissing(nlsw.grade) .>= 16)
    else
        @warn "Could not find collgrad or grade in: $(names(nlsw))"
        missing
    end

    if !ismissing(pct_never)
        println("% never married = ", round(pct_never*100; digits=2))
    end
    if !ismissing(pct_coll)
        println("% college grads = ", round(pct_coll*100; digits=2))
    end



    # (c) race proportions
    @assert rac !== nothing "race column not found"
    fr_race = freqtable(nlsw, rac)
    fr_race_prop = fr_race ./ sum(fr_race)
    println("race distribution (prop):\n", fr_race_prop)

    # (d) summary stats + missing grade
    summarystats = describe(nlsw, :mean, :median, :std, :min, :max, :nunique)
    @assert grd !== nothing "grade column not found"
    nmiss_grade = sum(ismissing, nlsw[!, grd])
    println("missing grade obs = ", nmiss_grade)
    CSV.write("nlsw88_summarystats.csv", summarystats)

    # (e) industry x occupation
    @assert ind !== nothing "industry column not found"
    @assert occ !== nothing "occupation column not found"
    jt = freqtable(nlsw, ind, occ)
    println("industry x occupation:\n", jt)

    # (f) mean wage by (industry, occupation)
    @assert wag !== nothing "wage column not found"
    df_w  = select(nlsw, ind, occ, wag)
    names!(df_w, [:industry, :occupation, :wage])
    df_grp = combine(groupby(df_w, [:industry, :occupation]), :wage => mean => :wage_mean)
    CSV.write("mean_wage_by_ind_occ.csv", df_grp)

    CSV.write("nlsw88_processed.csv", nlsw)
    return nothing
end


# ---------- q4 ----------
"""
matrixops(A,B): returns (A∘B, A'B, sum(A+B)). Inputs must have equal size.
"""
function matrixops(A::AbstractArray, B::AbstractArray)
    # Computes (A∘B), A'B, and sum(A+B); inputs must be same size.
    if size(A) != size(B)
        error("inputs must have the same size.")
    end
    hadamard  = A .* B
    crossprod = transpose(A) * B
    total_sum = sum(A .+ B)
    return hadamard, crossprod, total_sum
end

function q4()
    @assert isfile("firstmatrix.jld") "Run q1() first."
    d = JLD.load("firstmatrix.jld")
    A,B,C,D = d["A"], d["B"], d["C"], d["D"]

    hAB, cpAB, sAB = matrixops(A,B)
    println("matrixops(A,B): sum = ", sAB, " ; hadamard size = ", size(hAB))

    try
        matrixops(C,D)
    catch err
        println("matrixops(C,D) correctly errored: ", err)
    end

    @assert isfile("nlsw88_processed.csv") "Run q3() first for ttl_exp/wage."
    nlsw = CSV.read("nlsw88_processed.csv", DataFrame)
    @assert hascol(nlsw,:ttl_exp) "ttl_exp not found"; @assert hascol(nlsw,:wage) "wage not found"

    v1 = collect(skipmissing(nlsw.ttl_exp))
    v2 = collect(skipmissing(nlsw.wage))
    n  = min(length(v1), length(v2))
    V1, V2 = reshape(v1[1:n], n, 1), reshape(v2[1:n], n, 1)
    hVV, cpVV, sVV = matrixops(V1,V2)
    println("matrixops(ttl_exp,wage): sum = ", sVV, " ; hadamard size = ", size(hVV))
    return nothing
end

# ---------- q5: minimal tests ----------
@testset "PS1 sanity" begin
    A,B,C,D = q1()
    @test size(A)==(10,7) && size(B)==(10,7)
    @test size(C)==(5,7)  && size(D)==(10,7)
    @test all((A .<= 0) .== (D .== A))

    q2(A,B,C)
    @test isdefined(Main, :_Q2_cache)
    @test size(_Q2_cache.X) == (15169,6,5)
    @test size(_Q2_cache.β) == (6,5)
    @test size(_Q2_cache.Y) == (15169,5)

    if isfile("nlsw88.csv")
        q3()
        @test isfile("nlsw88_processed.csv")
        @test isfile("nlsw88_summarystats.csv")
        @test isfile("mean_wage_by_ind_occ.csv")
    end

    q4()
end

# ---------- entrypoint ----------
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n--- Running PS1 pipeline ---")
    A,B,C,D = q1()
    q2(A,B,C)
    if isfile("nlsw88.csv")
        q3()
    else
        println("nlsw88.csv not found; skipping q3()")
    end
    if isfile("firstmatrix.jld") && isfile("nlsw88.csv")
        q4()
    end
    println("Done. Artifacts written to ", pwd())
end
