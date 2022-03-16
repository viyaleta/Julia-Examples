# Super simple convolutional neural network with Julia Flux
# to predict hand-written digits with MNIST library

using Flux, Plots, ProgressMeter, MLDatasets;
using Flux: train!;
using Flux: onehotbatch;
using Flux: MaxPool;
using Colors;
using BSON: @save
gr();

# load full training set
train_x, train_y = MNIST.traindata()

train_x = Flux.unsqueeze(train_x, 3)  # add channels
train_y = onehotbatch(train_y, 0:9)

# load full test set
test_x,  test_y  = MNIST.testdata()

test_x = Flux.unsqueeze(test_x, 3)
test_y = onehotbatch(test_y, 0:9)

# to show image
# using Colors
# plot(Gray.(transpose(train_x[:, :, 1])))

# let's do a simple prediction neural network

network = Chain(
    Conv((5, 5), 1 => 16, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 16 => 32, relu),
    MaxPool((2, 2)),
    flatten,
    Dense(512, 10),
    softmax
)

parameters = Flux.params(network);

loss(x, y) = Flux.crossentropy(network(x), y);  # anon function

opt = ADAM(0.001);  # optimizer = gradient descent with learning rate

epochs = 10;

loss_history = Array{Float64}(undef, 0, 2)  # one column for train data, one for validation

train_data = Flux.DataLoader((train_x, train_y); batchsize=128, shuffle=true)

@showprogress for i in 1:epochs
    train!(loss, parameters, train_data, opt)
    loss_history = [loss_history;
        [loss(train_x, train_y) loss(test_x, test_y)]]
end

plot(loss_history, labels=["train" "validation"])

@save "mnist_cnn.bson" network


# show an image of a random prediction
random_i = rand(1:10000)  # random value
plot(Gray.(transpose(test_x[:, :, 1, random_i])))  # plot the random image
actual_value = onecold(test_y[:, random_i], 0:9)
predicted_value = onecold(network(reshape(test_x[:, :, :, random_i],
    (28, 28, 1, 1))), 0:9)[1]
annotate!((1, 0), text(
    "actual value: $actual_value \npredicted value: $predicted_value ",
    :top, :right, :red))
