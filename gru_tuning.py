from gru_setup import *
from kerastuner.tuners import RandomSearch
from plot_keras_history import plot_history
from matplotlib import pyplot as plt

model = build_model(plot_model_arch=True)

# MODEL CONFIG
batch_size = 128
no_epochs = 25
validation_data = (x_val, t_val)
verbosity = 1

# MODEL TUNING
tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=1,
    executions_per_trial=1,
    directory="results/gru_tuning_dir",
    project_name="DNNs_for_network_flow_classification",
)
tuner.search(
    x_train,
    t_train,
    epochs=no_epochs,
    validation_data=(x_val, t_val),
)
tuner.search_space_summary()
best_model = tuner.get_best_models()[0]

# MODEL FITTING
model = best_model.fit(
    x_train,
    t_train,
    batch_size=batch_size,
    epochs=no_epochs,
    verbose=verbosity,
    validation_data=validation_data,
)

# MODEL TESTING
test_metric_names = model.metrics_names
test_scores = model.evaluate(x_test, t_test, verbose=2)
for idx, score in enumerate(test_scores):
    print(test_metric_names[idx], ": ", score)

plt.plot(model.history["loss"])
plt.plot(model.history["val_loss"])
plt.savefig(
    f"results/GRU_model_perf_{round(test_scores[3]*100)}%_{round(test_scores[4]*100)}%"
)
# show_history(history)
# plot_history(
#     model,
#     path=f"results/GRU_model_perf_{round(test_scores[3]*100)}%_{round(test_scores[4]*100)}%",
# )
# plt.close()
