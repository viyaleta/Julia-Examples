# let's make a vae!
using Flux;
using Flux: train!, flatten, mse;
using Plots, ProgressMeter, MLDatasets;
using Colors;
using BSON: @save
using Distributions;
gr();

# load full training set
train_x, train_y = MNIST.traindata()

train_x = Flux.unsqueeze(train_x, 3)  # add channels
# train_y = onehotbatch(train_y, 0:9)

# input and output for vae
x = flatten(train_x)

# flat_x = reshape(train_x, :, size(train_x, 3))
# flatten x into 28^2, 60000 shape

# load full test set
test_x,  test_y  = MNIST.testdata()

test_x = Flux.unsqueeze(test_x, 3)
# test_y = onehotbatch(test_y, 0:9)


x_validation = flatten(test_x)
# simple autoencoder

latent_dims = 2;

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = tuple(map(f -> f(x), m.paths))

encode = Chain(
    Dense(28^2, 256, sigmoid),
    Split(Dense(256, latent_dims), Dense(256, latent_dims))  # z_μ and z_logσ
)

function z(z_μ, z_logσ)
    # reparameterization trick: z = μ + σ ⊙ ϵ and σ = ℯ ^ (log(σ^2))
    z_μ .+ rand(Normal(0, 1), latent_dims) .* exp.(0.5 .* z_logσ)
end

decode = Chain(
    Dense(2, 256, sigmoid),
    Dense(256, 28^2, sigmoid),
)

# KL divergence
# Σ σ^2 + μ^2 - log(σ) - 1
loss_kl(z_μ, z_logσ) = 0.5 * sum(exp.(2 * z_logσ) + z_μ.^2 .- z_logσ .- 1)

# reconstruction loss
loss_reconstruct(x, x̂) = sum(mse.(x, x̂))

function loss(x)
    z_μ, z_logσ = encode(x)[1]
    encoded_z = z(z_μ, z_logσ)
    x̂ = decode(encoded_z)  # reconstruction
    mean(loss_kl(z_μ, z_logσ) + loss_reconstruct(x, x̂))
end

function reconstruct(x)
    z_μ, z_logσ = encode(x)[1]
    encoded_z = z(z_μ, z_logσ)
    x̂ = decode(encoded_z)
    # sigmoid.(x̂)
end

parameters = Flux.params(encode, decode)

opt = ADAM()  # optimizer = gradient descent with learning rate

epochs = 5

loss_history = Array{Float64}(undef, 0, 2)

train_data = Flux.DataLoader((x); batchsize=128, shuffle=true)

@showprogress for i in 1:epochs
    train!(loss, parameters, train_data, opt)
    loss_history = [
        loss_history;
        [loss(x[:, :100]) loss(x_validation[:, :50])]]
end

plot(loss_history, labels=["train" "validation"])

@save "vae.bson" vae

sample = x[:, rand(1:size(x)[end])]


vae_fn(inp) = decode(getlatentspace(inp)[end])

vae_fn(sample)

getlatentspace(sample)
