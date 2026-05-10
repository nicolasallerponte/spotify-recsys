# Function to compute loss
def compute_loss(A, W):
    AW = sparse_dense_multiply(A, W)
    loss = mean_square_error(AW)
    
    # L2 Regularization
    reg_L2 = LAMBDA_A * (l2_norm(W)^2) / 2.0
    
    # L1 Regularization
    reg_L1 = LAMBDA_B * l1_norm(W)
    
    return loss + reg_L2 + reg_L1

# Training step
@tf.function
def train_step():
    gradients = compute_gradients(compute_loss, A, W)
    update_parameters(optimizer, [W], gradients)

# Load sparse matrix and convert to tensor
A = convert_sparse_matrix_to_sparse_tensor(A)

# Initialize matrix W randomly
items = A.columns
W = initialize_matrix(rows=items, cols=items)

# Define optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    loss = train_step()
    if (epoch + 1) % 10 == 0:
        print("Epoch:", epoch + 1, "Loss:", loss)

trained_W = get_trained_weights(W)
