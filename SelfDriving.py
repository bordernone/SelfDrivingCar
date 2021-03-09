import tensorflow as tf
import numpy as np
from random import sample


# converts a list of tuples in 1D into matrix, a = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
def convert_to_matrix(data):
    value = []
    for tuple_ in data:
        temp = (np.asarray(tuple_).astype("float32")).tolist()
        value.append(temp)
    return np.array(value, dtype="float32")


def gather_single_along_axis(data, axes):
    assert tf.is_tensor(data)
    assert tf.is_tensor(axes)

    axes_numpy = axes.numpy().astype("int32")
    new = []
    for i, x in enumerate(axes_numpy):
        temp = [i, x]
        new.append(temp)
    return tf.gather_nd(data, new)


class CarModel:
    def __init__(self, input_size, output_size):
        inputs = tf.keras.Input(shape=(input_size,))
        hidden_layer1 = tf.keras.layers.Dense(30, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(output_size)(hidden_layer1)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.Huber()

    def learn(self, batch):
        for step, (x_learn_batch, y_learn_batch) in enumerate(batch):
            with tf.GradientTape() as tape:
                predictions = self.model(x_learn_batch)
                loss = self.loss_function(y_learn_batch, predictions)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        print(loss)

    def learn_modified(self, batch):
        for step, (x_learn_batch, y_learn_batch, action_batch) in enumerate(batch):
            with tf.GradientTape() as tape:
                predictions = self.model(x_learn_batch)
                predictions_for_action = gather_single_along_axis(predictions, action_batch)
                loss = self.loss_function(y_learn_batch, predictions_for_action)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def forward_propagate(self, inputs, numpy_array=False):
        if not numpy_array:
            # Expects a tensorflow tensor as input, also the input has to be batched!
            predictions = self.model(inputs, training=True)
        else:
            inputs = np.asmatrix(inputs)
            predictions = self.model(inputs, training=True)

        return predictions


class Replay:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity

    # states = (signal1, signal2, signal3, orientation, -orientation)
    def push(self, last_state, new_state, action_taken, reward_earned):
        event = (last_state, new_state, action_taken, reward_earned)
        if len(self.memory) == self.capacity:
            del self.memory[0]
        self.memory.append(event)

    def get_sample(self, size):
        return np.asarray(sample(self.memory, size), dtype=object)


class Dqn:
    def __init__(self, input_size, output_size, gamma):
        self.gamma = gamma
        self.model = CarModel(input_size, output_size)
        self.replay = Replay(100000)
        self.last_state = [0, 0, 0, 0, 0]
        self.last_action = 0

    # Inputs means state (signal1, signal2, signal3, orientation, -orientation)
    def select_action(self, inputs, numpy_array=False):
        q_values = self.model.forward_propagate(inputs, numpy_array)
        predictions = tf.keras.activations.softmax(q_values)

        # Assumes vector output, two dimensional
        predictions_numpy = predictions.numpy()[0]
        draw = np.random.rand()

        accumulate = 0
        for i, pred in enumerate(predictions_numpy):
            accumulate = accumulate + pred
            if draw <= accumulate:
                return i
        print("Failed!")

    # new_signal = [signal1, signal2, signal3, orientation, -orientation]
    def update(self, last_reward, new_signal):
        self.replay.push(self.last_state, new_signal, self.last_action, last_reward)
        new_action = self.select_action(new_signal, numpy_array=True)

        learn_batch_size = 100
        if len(self.replay.memory) > learn_batch_size:
            train_samples = np.asarray(self.replay.get_sample(learn_batch_size), dtype=object)

            last_state_batch = train_samples[:, 0]
            assert last_state_batch.shape == (learn_batch_size,)
            last_state_batch = convert_to_matrix(last_state_batch)

            new_state_batch = train_samples[:, 1]
            assert new_state_batch.shape == (learn_batch_size,)
            new_state_batch = convert_to_matrix(new_state_batch)

            action_taken_batch = train_samples[:, 2]
            assert action_taken_batch.shape == (learn_batch_size,)
            action_taken_batch = convert_to_matrix(action_taken_batch)

            reward_batch = train_samples[:, 3]
            assert reward_batch.shape == (learn_batch_size,)
            reward_batch = convert_to_matrix(reward_batch)

            new_outputs = self.model.forward_propagate(new_state_batch, numpy_array=True)
            new_outputs_max = tf.reduce_max(new_outputs, axis=[1])
            assert tf.shape(new_outputs_max) == (learn_batch_size,)
            targets = self.gamma * new_outputs_max + reward_batch

            dataset = tf.data.Dataset.from_tensor_slices((last_state_batch, targets.numpy(), action_taken_batch))
            dataset = dataset.batch(100)
            self.model.learn_modified(dataset)

        self.last_state = new_signal
        self.last_action = new_action
        return new_action

    def score(self):
        return 0

    def save(self):
        pass

    def load(self):
        pass
