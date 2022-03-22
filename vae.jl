# let's make a vae!
using Flux;
using Flux: train!, flatten, mse, binarycrossentropy;
using MLDatasets;
using Distributions;
using Plots, Colors;
using ProgressMeter;
using BSON: @save
gr();  # Plots backend

# load the data sets and process data
train_x, _ = MNIST.traindata()
train_x = Flux.unsqueeze(train_x, 3)  # add channels
train_x = flatten(train_x)  # flatten the data

test_x,  _  = MNIST.testdata()
test_x = Flux.unsqueeze(test_x, 3)  # add channels
test_x = flatten(test_x)  # flatten the data

# specify latent space dimensions
latent_dims = 2;

# custom split layer
struct Split{T}
  paths::T  # stores paths tuple (source layers in the model)
end

# function defined for Split layer that takes in  a tuple and stores it into the object
Split(paths...) = Split(paths)

# default action for input of abstract array
(m::Split)(x::AbstractArray) = tuple(map(f -> f(x), m.paths))

# make Split object callable
Flux.@functor Split

encode = Chain(
    Dense(28^2, 512, relu),
    Dense(512, 256, relu),
    Split(Dense(256, latent_dims, relu), Dense(256, latent_dims, relu))  # z_μ and z_logσ
)

function z(z_μ, z_logσ)
    # reparameterization trick: z = μ + σ ⊙ ϵ and σ = ℯ ^ (log(σ^2))
    z_μ .+ rand(Normal(0, 1), latent_dims) .* exp.(0.5 .* z_logσ)
end

decode = Chain(
    Dense(2, 256, relu),
    Dense(256, 512, relu),
    Dense(512, 28^2, relu),
)

# KL divergence
# Σ σ^2 + μ^2 - log(σ) - 1
loss_kl(z_μ, z_logσ) = 0.5 * sum(exp.(2 * z_logσ) + z_μ.^2 .- z_logσ .- 1)

# reconstruction loss
loss_reconstruct(x, x̂) = mse(x, x̂)

function loss(x)
    z_μ, z_logσ = encode(x)[1]
    encoded_z = z(z_μ, z_logσ)
    x̂ = decode(encoded_z)  # reconstruction
    mean(loss_kl(z_μ, z_logσ) + loss_reconstruct(x, x̂))
end

function reconstruct(x)
    z_μ, z_logσ = encode(x)[1]
    encoded_z = z(z_μ, z_logσ)
    decode(encoded_z)
    # sigmoid.(x̂)
end

parameters = Flux.params(encode, decode)

opt = ADAM(0.001)  # optimizer = gradient descent with learning rate

epochs = 10

loss_history = Array{Float64}(undef, 0, 2)

train_data = Flux.DataLoader((train_x[:, 1:1000]); batchsize=128, shuffle=true)

@showprogress for i in 1:epochs
    train!(loss, parameters, train_data, opt)
    loss_history = [
        loss_history;
        [loss(train_x[:, 1:1000]) loss(test_x[:, 1:1000])]]
end

plot(loss_history, labels=["train" "validation"])

@save "encode.bson" encode
@save "decode.bson" decode
