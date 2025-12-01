README — Weather Forecasting With Deep Learning

This project focuses on building a 7-day (168-hour) weather forecasting system using several deep learning architectures. All experiments were run on Modal, which provided GPU acceleration and persistent storage through Modal volumes.

The process began with a dedicated preprocessing script that transformed the raw weather dataset into a machine-learning-ready format. Each sample was constructed using a 10-day input window with hourly readings, giving models 240 timesteps of historical information. The goal was to predict the following 168 hours (seven days). All processed data (X.npy and Y.npy) was stored inside a Modal volume so that every model could access the same standardized dataset.

With the data prepared, three different neural network architectures were implemented and trained independently: an LSTM Seq2Seq model with attention, a Temporal Convolutional Neural Network (TCN), and a Transformer-based model. Each architecture had its own training script designed to load data directly from the shared Modal volume, train the model on an A100 GPU, and store all checkpoints and final weights back into the volume. This setup allowed the team to experiment with model-specific improvements while maintaining a consistent data pipeline.

For evaluation, each team member developed their own testing script tailored to their model to allow them to eveluate their model independently. Every script saved its resulting performance metrics and plots back into the shared Modal volume. After all individual evaluations were completed, a unified benchmarking script was created to test all models side-by-side. It compared LSTM, TCN, and Transformer architectures across several metrics.

Once the strongest architecture was identified, the team used a custom hyperparameter optimization (HPO) script to fine-tune the choosen model further. This process explored variations in learning rate, hidden sizes, dropout, sequence modeling parameters, teacher forcing, and other training hyperparameters. The best-performing tuned model was saved back into the volume for final use.

Finally, a dedicated plotting script was used to generate clean and comprehensive visualizations for the final optimized model.
