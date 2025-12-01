README — Weather Forecasting With Deep Learning

This project focuses on building a 7-day (168-hour) weather forecasting system using several deep learning architectures. All experiments were run on Modal, which provided GPU acceleration and persistent storage through Modal volumes.

The process began with a dedicated Data_Preprocessing.py that transformed the raw weather dataset into a machine-learning-ready format. Each sample was constructed using a 10-day input window with hourly readings, giving models 240 timesteps of historical information. The goal was to predict the following 168 hours (seven days). All processed data (X.npy and Y.npy) was stored inside a Modal volume so that every model could access the same standardized dataset for future training.

With the data prepared, three different neural network architectures were implemented and trained independently: an LSTM Seq2Seq model with attention (train_lstm.py), a Temporal Convolutional Neural Network (TCN) (train_tcnn.py), and a Transformer-based model (train_transformer.py). Each architecture had its own training script designed to load data directly from the shared Modal volume, train the model on an A100 GPU, and store all checkpoints and final weights back into the modal volume. This setup allowed the team to experiment with model-specific improvements without overwriting others training code.

For evaluation, each team member developed their own testing script tailored to their model to allow them to eveluate their model independently. Every script saved its resulting performance metrics and plots back into the shared Modal volume. Testing scripts were named: test_lstm.py, test_tcnn.py and test_transformer.py. After all individual evaluations were completed, a unified benchmarking script (compare_models.py) was created to test all models side-by-side. It compared LSTM, TCN, and Transformer architectures across several key metrics.

Once the strongest architecture was identified which was the FED Former, the team created a custom hyperparameter optimization (HPO) script to fine-tune the best model further. This process explored variations in learning rate, batch size, dropout and weight decay. The final tuned weights of the model was saved back into the volume for final use.

Finally, the compare_models.py plotting script was used to generate clean and comprehensive visualizations for the final optimized model in comparison to the three previous models.
