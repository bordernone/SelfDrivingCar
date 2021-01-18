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


# if __name__ == "__main__":
#     dqn = Dqn(5, 3, 0.9)
#     some_data = []
#     some_data.append(((1, 1, 1, 1, 1), (2, 2, 2, 2, 2), 0, 0))
#     some_data.append(((1, 1, 1, 1, 1), (2, 2, 2, 2, 2), 2, 1))
#     some_data.append(((1, 1, 1, 1, 1), (2, 2, 2, 2, 2), 1, 2))
#     some_data.append(((1, 1, 1, 1, 1), (2, 2, 2, 2, 2), 2, 0))
#
#     some_data = np.asarray(some_data, dtype=object)
#     train_samples = some_data
#     learn_batch_size = 4
#     gamma = 0.9
#
#     # last_state_batch = train_samples[:, 0]
#     # assert last_state_batch.shape == (learn_batch_size,)
#     # last_state_batch = convert_to_matrix(last_state_batch)
#     #
#     # new_state_batch = train_samples[:, 1]
#     # assert new_state_batch.shape == (learn_batch_size,)
#     # new_state_batch = convert_to_matrix(new_state_batch)
#     #
#     # action_taken_batch = train_samples[:, 2]
#     # assert action_taken_batch.shape == (learn_batch_size,)
#     # action_taken_batch = convert_to_matrix(action_taken_batch)
#     #
#     # reward_batch = train_samples[:, 3]
#     # assert reward_batch.shape == (learn_batch_size,)
#     # reward_batch = convert_to_matrix(reward_batch)
#     #
#     # new_outputs = dqn.model.forward_propagate(new_state_batch, numpy_array=True)
#     # new_outputs_max = tf.reduce_max(new_outputs, axis=[1])
#     # assert tf.shape(new_outputs_max) == (learn_batch_size,)
#     # targets = gamma * new_outputs_max + reward_batch
#     #
#     # dataset = tf.data.Dataset.from_tensor_slices((last_state_batch, targets.numpy(), action_taken_batch))
#     # dataset = dataset.batch(1)
#     # dqn.model.learn_modified(dataset)
#
#     x = dqn.model.forward_propagate(np.asmatrix([[0., 0., 0., -0.9075264, 0.9075264],
#                                                  [0., 0., 0., -0.79438114, 0.79438114],
#                                                  [0., 0., 0., -0.92994285, 0.92994285],
#                                                  [0., 0., 0., -0.92907417, 0.92907417],
#                                                  [0., 0., 0., 0.9811021, -0.9811021],
#                                                  [0., 0., 0., -0.71021956, 0.71021956],
#                                                  [0., 0., 0., -0.688942, 0.688942],
#                                                  [0., 0., 0., -0.7125726, 0.7125726],
#                                                  [0., 0., 0., -0.81501853, 0.81501853],
#                                                  [0., 0., 0., -0.9084032, 0.9084032],
#                                                  [0., 0., 0., -0.81939924, 0.81939924],
#                                                  [0., 0., 0., -0.92549753, 0.92549753],
#                                                  [0., 0., 0., -0.9197744, 0.9197744],
#                                                  [0., 0., 0., -0.80776167, 0.80776167],
#                                                  [0., 0., 0., -0.9167216, 0.9167216],
#                                                  [0., 0., 0., 0.9790075, -0.9790075],
#                                                  [0., 0., 0., -0.80278045, 0.80278045],
#                                                  [0., 0., 0., 0.95703477, -0.95703477],
#                                                  [0., 0., 0., -0.90972275, 0.90972275],
#                                                  [0., 0., 0., -0.93120086, 0.93120086],
#                                                  [0., 0., 0., -0.80994564, 0.80994564],
#                                                  [0., 0., 0., 0.8515011, -0.8515011],
#                                                  [0., 0., 0., -0.9296556, 0.9296556],
#                                                  [0., 0., 0., -0.797108, 0.797108],
#                                                  [0., 0., 0., -0.929961, 0.929961],
#                                                  [0., 0., 0., 0.9817615, -0.9817615],
#                                                  [0., 0., 0., -0.81207365, 0.81207365],
#                                                  [0., 0., 0., -0.7931922, 0.7931922],
#                                                  [0., 0., 0., 0.98216915, -0.98216915],
#                                                  [0., 0., 0., -0.9175537, 0.9175537],
#                                                  [0., 0., 0., 0.98301154, -0.98301154],
#                                                  [0., 0., 0., 0.7373363, -0.7373363],
#                                                  [0., 0., 0., 0.87118834, -0.87118834],
#                                                  [0., 0., 0., -0.70182383, 0.70182383],
#                                                  [0., 0., 0., -0.90925306, 0.90925306],
#                                                  [0., 0., 0., -0.9171397, 0.9171397],
#                                                  [0., 0., 0., -0.9093504, 0.9093504],
#                                                  [0., 0., 0., 0.8478042, -0.8478042],
#                                                  [0., 0., 0., -0.9066212, 0.9066212],
#                                                  [0., 0., 0., 0.9570655, -0.9570655],
#                                                  [0., 0., 0., -0.8090573, 0.8090573],
#                                                  [0., 0., 0., -0.80373174, 0.80373174],
#                                                  [0., 0., 0., -0.8223396, 0.8223396],
#                                                  [0., 0., 0., -0.7990766, 0.7990766],
#                                                  [0., 0., 0., -0.92835015, 0.92835015],
#                                                  [0., 0., 0., 0.73937815, -0.73937815],
#                                                  [0., 0., 0., 0.96323293, -0.96323293],
#                                                  [0., 0., 0., 0.9571463, -0.9571463],
#                                                  [0., 0., 0., -0.93022776, 0.93022776],
#                                                  [0., 0., 0., -0.8168941, 0.8168941],
#                                                  [0., 0., 0., 0.8688904, -0.8688904],
#                                                  [0., 0., 0., -0.9315512, 0.9315512],
#                                                  [0., 0., 0., -0.9315135, 0.9315135],
#                                                  [0., 0., 0., -0.9072314, 0.9072314],
#                                                  [0., 0., 0., -0.9287799, 0.9287799],
#                                                  [0., 0., 0., -0.80467504, 0.80467504],
#                                                  [0., 0., 0., -0.9088845, 0.9088845],
#                                                  [0., 0., 0., -0.8112054, 0.8112054],
#                                                  [0., 0., 0., -0.80685264, 0.80685264],
#                                                  [0., 0., 0., -0.81771183, 0.81771183],
#                                                  [0., 0., 0., -0.69030964, 0.69030964],
#                                                  [0., 0., 0., -0.9312065, 0.9312065],
#                                                  [0., 0., 0., -0.9296342, 0.9296342],
#                                                  [0., 0., 0., -0.9066695, 0.9066695],
#                                                  [0., 0., 0., 0.9569532, -0.9569532],
#                                                  [0., 0., 0., -0.8206466, 0.8206466],
#                                                  [0., 0., 0., 0.86974144, -0.86974144],
#                                                  [0., 0., 0., -0.7135665, 0.7135665],
#                                                  [0., 0., 0., -0.8153763, 0.8153763],
#                                                  [0., 0., 0., -0.8165041, 0.8165041],
#                                                  [0., 0., 0., -0.9057814, 0.9057814],
#                                                  [0., 0., 0., -0.9300933, 0.9300933],
#                                                  [0., 0., 0., -0.93162906, 0.93162906],
#                                                  [0., 0., 0., -0.81417775, 0.81417775],
#                                                  [0., 0., 0., 0.9599418, -0.9599418],
#                                                  [0., 0., 0., -0.9062059, 0.9062059],
#                                                  [0., 0., 0., 0.8461138, -0.8461138],
#                                                  [0., 0., 0., -0.9258149, 0.9258149],
#                                                  [0., 0., 0., 0.9586041, -0.9586041],
#                                                  [0., 0., 0., -0.9219378, 0.9219378],
#                                                  [0., 0., 0., 0.73565704, -0.73565704],
#                                                  [0., 0., 0., 0.8680202, -0.8680202],
#                                                  [0., 0., 0., 0.7383603, -0.7383603],
#                                                  [0., 0., 0., -0.815764, 0.815764],
#                                                  [0., 0., 0., -0.9070277, 0.9070277],
#                                                  [0., 0., 0., -0.90767276, 0.90767276],
#                                                  [0., 0., 0., 0.97952795, -0.97952795],
#                                                  [0., 0., 0., -0.92936605, 0.92936605],
#                                                  [0., 0., 0., -0.92864984, 0.92864984],
#                                                  [0., 0., 0., -0.9314832, 0.9314832],
#                                                  [0., 0., 0., 0.98074144, -0.98074144],
#                                                  [0., 0., 0., 0.9825857, -0.9825857],
#                                                  [0., 0., 0., -0.92766523, 0.92766523],
#                                                  [0., 0., 0., -0.9273221, 0.9273221],
#                                                  [0., 0., 0., 0.8704687, -0.8704687],
#                                                  [0., 0., 0., 0.98086905, -0.98086905],
#                                                  [0., 0., 0., -0.9261357, 0.9261357],
#                                                  [0., 0., 0., 0.9798842, -0.9798842],
#                                                  [0., 0., 0., -0.81930244, 0.81930244],
#                                                  [0., 0., 0., -0.9079169, 0.9079169]]).astype("float32"))
#     print(x)
