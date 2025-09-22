from req_prog.architectures import UnitaryModel
from req_prog.training import train_model
from req_prog.training_h import train_model_h
from req_prog.useful_functions import reset_weights

# Set the parameters for the training, that is, the number of qubits for the models and the delta_t related with the dataframe used for train the model
Ns = [n for n in range(2, 9)]
delta_t = [0.1, 0.25, 0.5, 1.0]
method = "trotter"

# Start the training for multiple models with multiple datasets
for N_qbits in Ns:
    for dt in delta_t:
        model = UnitaryModel(n_qubits = N_qbits)
        model.apply(reset_weights)

        if N_qbits in [4, 5, 6, 7, 8]:
            device = "cuda"
        else:
            device = "cpu"

        if N_qbits <= 6:
            train_model(
                model = model,
                N_qbits = N_qbits,
                delta_t = dt,
                num_epochs = 1000,
                batch_size = 10,
                batch_epoch = 100,
                learning_rate = 1e-3,
                N_domain = 100,
                sch = 300,
                device = device
                )
            
        else:
            train_model_h(
                model = model,
                N_qbits = N_qbits,
                delta_t = dt,
                sch = 300,
                device = device,
                num_epochs = 1000,
                batch_size = 10,
                batch_epoch = 10,
                method = method
                )