# let's make a vae!
using Flux;
using Flux: train!, onehotbatch, MaxPool, flatten;
using Plots, ProgressMeter, MLDatasets;
using Colors;
using BSON: @save
gr();

# load full training set
train_x, train_y = MNIST.traindata()

train_x = Flux.unsqueeze(train_x, 3)  # add channels
train_y = onehotbatch(train_y, 0:9)

# input and output for vae
x = flatten(train_x)

# flat_x = reshape(train_x, :, size(train_x, 3))
# flatten x into 28^2, 60000 shape

# load full test set
test_x,  test_y  = MNIST.testdata()

test_x = Flux.unsqueeze(test_x, 3)
test_y = onehotbatch(test_y, 0:9)


x_validation = flatten(test_x)
# simple autoencoder

encode = Chain(
    Dense(28^2, 500, relu),
    Dense(500, 250, relu),
    Dense(250, 2, relu),
)

decode = Chain(
    Dense(2, 250, relu),
    Dense(250, 500, relu),
    Dense(500, 28^2, relu),
)

vae = Chain(encode, decode)

parameters = Flux.params(vae);

loss(x_in, x_out) = Flux.mse(vae(x_in), x_out);  # anon function

opt = ADAM(0.001);  # optimizer = gradient descent with learning rate

epochs = 2;

loss_history = Array{Float64}(undef, 0, 2)

train_data = Flux.DataLoader((x, x); batchsize=128, shuffle=true)

@showprogress for i in 1:epochs
    train!(loss, parameters, train_data, opt)
    loss_history = [loss_history;
        [loss(x, x) loss(x_validation, x_validation)]]
end

plot(loss_history, labels=["train" "validation"])

@save "vae.bson" vae
