#could be more general but i dont care
struct col
    m::Matrix
end

struct line
    m::Matrix
end

function iterate(m::col, state::Int64 = 1) #stop when out of bound exception
    try
        return m.m[:, state], state+1
    catch
        return nothing
    end
end


function iterate(m::line, state::Int64 = 1) #stop when out of bound exception
    try
        return m.m[state, :], state+1
    catch
        return nothing
    end
end

function eye(m::Int64)
    return Array{Float64, 2}(I, m, m)
end