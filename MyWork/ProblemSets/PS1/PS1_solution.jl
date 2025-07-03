#=
Problem Set 1 - Julia Introduction
Course: ECON 6343: Econometrics III
Professor: Tyler Ransom
University of Oklahoma

Group Members:
- Bhaskar Kaushik

Date: August 2024
=#

# Load required packages
using JLD
using Random
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using FreqTables
using Distributions

# Set random seed
Random.seed!(1234)

#==============================================================================
Question 1: Initializing variables and practice with basic matrix operations
==============================================================================#

function q1()
    # Set seed for reproducibility
    Random.seed!(1234)
    
    # Part (a): Create matrices
    
    # i. A: 10×7 matrix of random numbers distributed U[-5,10]
    A = rand(10, 7) .* 15 .- 5  # Transform U[0,1] to U[-5,10]
    
    # ii. B: 10×7 matrix of random numbers distributed N(-2,15) [std dev is 15]
    B = randn(10, 7) .* 15 .- 2
    
    # iii. C: 5×7 matrix - first 5 rows and first 5 columns of A, and last 2 columns of B
    C = hcat(A[1:5, 1:5], B[1:5, 6:7])
    
    # iv. D: 10×7 matrix where D[i,j] = A[i,j] if A[i,j] ≤ 0, or 0 otherwise
    D = A .* (A .<= 0)
    
    # Part (b): Number of elements in A
    println("Number of elements in A: ", length(A))
    
    # Part (c): Number of unique elements in D
    println("Number of unique elements in D: ", length(unique(D)))
    
    # Part (d): Create E using reshape (vec operator applied to B)
    E = reshape(B, :, 1)  # This is equivalent to vec(B)
    # Easier way:
    E_easier = vec(B)
    println("E and E_easier are equal: ", E == E_easier)
    
    # Part (e): Create 3D array F
    F = zeros(10, 7, 2)
    F[:, :, 1] = A
    F[:, :, 2] = B
    
    # Part (f): Permute dimensions of F
    F = permutedims(F, (3, 1, 2))  # Now F is 2×10×7
    println("New dimensions of F: ", size(F))
    
    # Part (g): Kronecker product
    G = kron(B, C)
    println("Dimensions of G (B⊗C): ", size(G))
    
    # Try C⊗F (this should produce an error or unexpected result due to incompatible dimensions)
    try
        result = kron(C, F)
        println("C⊗F worked, dimensions: ", size(result))
    catch e
        println("C⊗F failed with error: ", e)
    end
    
    # Part (h): Save matrices as .jld file
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)
    
    # Part (i): Save only A, B, C, D
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)
    
    # Part (j): Export C as CSV
    C_df = DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", C_df)
    
    # Part (k): Export D as tab-delimited .dat file
    D_df = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", D_df; delim='\t')
    
    # Part (l): Return A, B, C, D
    return A, B, C, D
end

#==============================================================================
Question 2: Practice with loops and comprehensions
==============================================================================#

function q2(A, B, C)
    # Part (a): Element-by-element product of A and B
    # Using comprehension
    AB = [A[i,j] * B[i,j] for i in 1:size(A,1), j in 1:size(A,2)]
    
    # Without loop or comprehension
    AB2 = A .* B
    
    println("AB and AB2 are equal: ", AB ≈ AB2)
    
    # Part (b): Elements of C between -5 and 5 (inclusive)
    # Using loop
    Cprime = Float64[]
    for i in 1:size(C,1)
        for j in 1:size(C,2)
            if -5 <= C[i,j] <= 5
                push!(Cprime, C[i,j])
            end
        end
    end
    
    # Without loop
    Cprime2 = vec(C[-5 .<= C .<= 5])
    
    println("Cprime and Cprime2 have same length: ", length(Cprime) == length(Cprime2))
    
    # Part (c): Create 3D array X
    N = 15_169
    K = 6
    T = 5
    
    # Initialize X
    X = zeros(N, K, T)
    
    # Fill X for each time period
    for t in 1:T
        # Column 1: Intercept
        X[:, 1, t] .= 1
        
        # Column 2: Dummy variable with probability 0.75*(6-t)/5
        prob = 0.75 * (6 - t) / 5
        X[:, 2, t] = rand(N) .< prob
        
        # Column 3: Normal with mean 15+t-1 and std dev 5(t-1)
        X[:, 3, t] = randn(N) .* (5 * (t - 1)) .+ (15 + t - 1)
        
        # Column 4: Normal with mean sqrt((6-t)/3) and std dev 1/e
        X[:, 4, t] = randn(N) ./ ℯ .+ sqrt((6 - t) / 3)
        
        # Column 5: Binomial(20, 0.6) - "discrete normal"
        X[:, 5, t] = rand(Binomial(20, 0.6), N)
        
        # Column 6: Binomial(20, 0.5)
        X[:, 6, t] = rand(Binomial(20, 0.5), N)
    end
    
    # Part (d): Create matrix β
    β = zeros(K, T)
    for t in 1:T
        β[1, t] = 1 + 0.25 * (t - 1)    # 1, 1.25, 1.5, ...
        β[2, t] = log(t)                 # ln(t)
        β[3, t] = -sqrt(t)               # -√t
        β[4, t] = exp(t) - exp(t + 1)   # e^t - e^(t+1)
        β[5, t] = t                      # t
        β[6, t] = t / 3                  # t/3
    end
    
    # Part (e): Create matrix Y
    Y = zeros(N, T)
    σ = 0.36
    
    for t in 1:T
        ε = randn(N) .* σ
        Y[:, t] = X[:, :, t] * β[:, t] + ε
    end
    
    println("Dimensions of X: ", size(X))
    println("Dimensions of β: ", size(β))
    println("Dimensions of Y: ", size(Y))
    
    # Part (f): Function returns nothing
    return nothing
end

#==============================================================================
Question 3: Reading in Data and calculating summary statistics
==============================================================================#

function q3()
    # Part (a): Import nlsw88.csv
    nlsw88 = CSV.read("nlsw88.csv", DataFrame; missingstring="")
    
    # Save as processed file
    CSV.write("nlsw88_processed.csv", nlsw88)
    
    # Part (b): Percentage never married and college graduates
    # Never married
    never_married_pct = mean(skipmissing(nlsw88.never_married)) * 100
    println("\nPercentage never married: ", round(never_married_pct, digits=2), "%")
    
    # College graduates
    college_grad_pct = mean(skipmissing(nlsw88.collgrad)) * 100
    println("Percentage college graduates: ", round(college_grad_pct, digits=2), "%")
    
    # Part (c): Race distribution
    println("\nRace distribution:")
    race_freq = freqtable(nlsw88.race)
    race_pct = prop(race_freq) * 100
    for (race, pct) in enumerate(race_pct)
        println("  Race $race: $(round(pct, digits=2))%")
    end
    
    # Part (d): Summary statistics
    println("\nSummary statistics:")
    summarystats = describe(nlsw88)
    println(summarystats)
    
    # Count missing grade observations
    grade_missing = sum(ismissing.(nlsw88.grade))
    println("\nNumber of missing grade observations: ", grade_missing)
    
    # Part (e): Joint distribution of industry and occupation
    println("\nJoint distribution of industry and occupation:")
    # Remove missing values for crosstab
    df_ind_occ = dropmissing(nlsw88, [:industry, :occupation])
    cross_tab = freqtable(df_ind_occ.industry, df_ind_occ.occupation)
    println(cross_tab)
    
    # Part (f): Mean wage by industry and occupation
    println("\nMean wage by industry and occupation:")
    
    # Subset to relevant columns and remove missing
    wage_data = select(nlsw88, :industry, :occupation, :wage)
    wage_data = dropmissing(wage_data)
    
    # Group by industry and occupation, calculate mean wage
    wage_by_ind_occ = combine(groupby(wage_data, [:industry, :occupation]), 
                              :wage => mean => :mean_wage)
    
    # Sort for better readability
    sort!(wage_by_ind_occ, [:industry, :occupation])
    println(wage_by_ind_occ)
    
    return nothing
end

#==============================================================================
Question 4: Practice with functions
==============================================================================#

function q4()
    # Part (a): Load firstmatrix.jld
    data = load("firstmatrix.jld")
    A = data["A"]
    B = data["B"]
    C = data["C"]
    D = data["D"]
    
    # Parts (b) and (c): Define matrixops function
    """
    matrixops - Performs matrix operations on two input matrices
    
    This function takes two matrices A and B as inputs and returns:
    1. The element-by-element product of A and B
    2. The matrix product A'B (A transpose times B)
    3. The sum of all elements in A+B
    
    Inputs must be matrices of the same size.
    """
    function matrixops(A, B)
        # Part (e): Check if inputs have the same size
        if size(A) != size(B)
            error("inputs must have the same size.")
        end
        
        # (i) Element-by-element product
        elem_product = A .* B
        
        # (ii) Matrix product A'B
        matrix_product = A' * B
        
        # (iii) Sum of all elements of A+B
        sum_all = sum(A + B)
        
        return elem_product, matrix_product, sum_all
    end
    
    # Part (d): Evaluate matrixops with A and B
    println("\nEvaluating matrixops with A and B:")
    elem_prod, mat_prod, sum_AB = matrixops(A, B)
    println("  Element-wise product size: ", size(elem_prod))
    println("  A'B product size: ", size(mat_prod))
    println("  Sum of all elements in A+B: ", sum_AB)
    
    # Part (f): Try with C and D (should error)
    println("\nTrying matrixops with C and D:")
    try
        matrixops(C, D)
    catch e
        println("  Error as expected: ", e)
    end
    
    # Part (g): Try with wage and ttl_exp from the data
    println("\nTrying matrixops with wage and ttl_exp from nlsw88:")
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    
    # Convert to arrays and handle missing values
    wage_array = collect(skipmissing(nlsw88.wage))
    ttl_exp_array = collect(skipmissing(nlsw88.ttl_exp))
    
    # Make sure they have the same length
    min_length = min(length(wage_array), length(ttl_exp_array))
    wage_array = wage_array[1:min_length]
    ttl_exp_array = ttl_exp_array[1:min_length]
    
    # Reshape to column vectors for matrix operations
    wage_matrix = reshape(wage_array, :, 1)
    ttl_exp_matrix = reshape(ttl_exp_array, :, 1)
    
    try
        elem_prod2, mat_prod2, sum_wage_exp = matrixops(wage_matrix, ttl_exp_matrix)
        println("  Success! Sum of wage + ttl_exp: ", sum_wage_exp)
    catch e
        println("  Error: ", e)
    end
    
    return nothing
end

#==============================================================================
Question 5: Unit tests
==============================================================================#

# Unit tests for q1
function test_q1()
    println("\n" * "="^60)
    println("Running unit tests for q1()...")
    
    A, B, C, D = q1()
    
    # Test dimensions
    @assert size(A) == (10, 7) "A should be 10×7"
    @assert size(B) == (10, 7) "B should be 10×7"
    @assert size(C) == (5, 7) "C should be 5×7"
    @assert size(D) == (10, 7) "D should be 10×7"
    
    # Test that A is in range [-5, 10]
    @assert all(-5 .<= A .<= 10) "A should be in range [-5, 10]"
    
    # Test that D has correct properties
    @assert all((D .== 0) .| (D .== A)) "D should be 0 or equal to A"
    @assert all(D[A .> 0] .== 0) "D should be 0 where A > 0"
    
    # Test file creation
    @assert isfile("matrixpractice.jld") "matrixpractice.jld should exist"
    @assert isfile("firstmatrix.jld") "firstmatrix.jld should exist"
    @assert isfile("Cmatrix.csv") "Cmatrix.csv should exist"
    @assert isfile("Dmatrix.dat") "Dmatrix.dat should exist"
    
    println("✓ All tests for q1() passed!")
end

# Unit tests for q2
function test_q2()
    println("\n" * "="^60)
    println("Running unit tests for q2()...")
    
    # Create test matrices
    Random.seed!(1234)
    A_test = rand(3, 3)
    B_test = rand(3, 3)
    C_test = randn(3, 3)
    
    # q2 should return nothing
    result = q2(A_test, B_test, C_test)
    @assert result === nothing "q2 should return nothing"
    
    println("✓ All tests for q2() passed!")
end

# Unit tests for q3
function test_q3()
    println("\n" * "="^60)
    println("Running unit tests for q3()...")
    
    # Ensure data file exists
    @assert isfile("nlsw88.csv") "nlsw88.csv should exist"
    
    # Run q3
    result = q3()
    @assert result === nothing "q3 should return nothing"
    
    # Check output file
    @assert isfile("nlsw88_processed.csv") "nlsw88_processed.csv should exist"
    
    println("✓ All tests for q3() passed!")
end

# Unit tests for q4
function test_q4()
    println("\n" * "="^60)
    println("Running unit tests for q4()...")
    
    # Ensure required file exists
    @assert isfile("firstmatrix.jld") "firstmatrix.jld should exist"
    
    # Run q4
    result = q4()
    @assert result === nothing "q4 should return nothing"
    
    println("✓ All tests for q4() passed!")
end

#==============================================================================
Main execution
==============================================================================#

println("="^80)
println("ECON 6343 - Problem Set 1: Julia Introduction")
println("Student: Bhaskar Kaushik")
println("="^80)

# Execute all functions in order
println("\n" * "="^60)
println("Executing Question 1...")
A, B, C, D = q1()

println("\n" * "="^60)
println("Executing Question 2...")
q2(A, B, C)

println("\n" * "="^60)
println("Executing Question 3...")
q3()

println("\n" * "="^60)
println("Executing Question 4...")
q4()

# Run unit tests
println("\n" * "="^80)
println("RUNNING UNIT TESTS")
println("="^80)

test_q1()
test_q2()
test_q3()
test_q4()

println("\n" * "="^80)
println("Problem Set 1 completed successfully!")
println("All tests passed! ✓")
println("="^80)
