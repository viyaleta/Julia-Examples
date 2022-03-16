using Random, Flux, Distributions, Plots;
using Flux: train!
gr();  # backend
using ProgressMeter;
using BSON: @save

# Make some data
x = rand(1:10, 2, 100)
y = sum(eachrow(x));
y = reshape(y, 1, length(y));

n_features = size(x)[1];
output_dim = size(y)[1];

# train / test split
idx_train = sample(1:size(x)[2], Int(0.7*size(x)[2]); replace=false);
idx_test = [i for i in collect(1:100) if i ∉ idx_train];

x_train = x[:, idx_train]
y_train = y[:, idx_train]
x_test = x[:, idx_test]
y_test = y[:, idx_test]

# for fun...
histogram(x_train[1, :]; alpha=0.5, nbins=10)
histogram!(x_test[1, :]; alpha=0.5, nbins=10)

# let's build the model
model_data = Flux.DataLoader((x_train, y_train); batchsize=10, shuffle=true)

model = Chain(
    Dense(n_features, 10, sigmoid),
    Dense(10, output_dim))

loss(x, y) = Flux.mse(model(x), y);  # anon function

opt = Descent(0.001);  # optimizer = gradient descent with learning rate

parameters = Flux.params(model);

# train

epochs = 1000;

loss_history = Array{Float64}(undef, 0, 2)

@showprogress for i in 1:epochs
    train!(loss, parameters, model_data, opt)
    loss_history = [loss_history; [loss(x_train, y_train) loss(x_test, y_test)]]
end

plot(loss_history, labels=["train" "validation"])


# let's try a different optimizer

opt = ADAGrad(0.1)

epochs = 1000;

loss_history = Array{Float64}(undef, 0, 2)

@showprogress for i in 1:epochs
    train!(loss, parameters, model_data, opt)
    loss_history = [loss_history; [loss(x_train, y_train) loss(x_test, y_test)]]
end

plot(loss_history, labels=["train" "validation"])
