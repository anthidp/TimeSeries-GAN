"""
Import packages & dataset

"""
generator_results = []
discriminator_results = []
generator_weights = []
discriminator_weights = []
checkpoint_path = "..."
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 64 
epochs = 100  


def features_transformation(train, seq_length, seq_step, num_signals):
""" 
Features reshape to time windows  & reduction with PCA. 


"""
    m, n = train.shape  # m=562387, n=35
    # normalization
    for i in range(n - 1):
        # print('i=', i)
        A = max(train[:, i])
        # print('A=', A)
        if A != 0:
            train[:, i] /= max(train[:, i])
            # scale from -1 to 1
            train[:, i] = 2 * train[:, i] - 1
        else:
            train[:, i] = train[:, i]

    samples = train[:, 0:n - 1]
    labels = train[:, n - 1]  # the last colummn is label
    #############################
    ############################
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    from sklearn.decomposition import PCA
    X_n = samples
    ####################################
    ###################################
    n_components = num_signals
    pca = PCA(n_components , svd_solver='full')  
    pca.fit(X_n)
    ex_var = pca.explained_variance_ratio_
    pc = pca.components_
   # num_signals = pc.shape[1] - 1
    # projected values on the principal component
    T_n = np.matmul(X_n, pc.transpose(1, 0))
    samples = T_n
    num_samples = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb

    return samples, labels


generator = keras.Sequential(
    [
        keras.Input(shape=(30,68), name="fake_input"),
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        layers.LSTM(100, name='first_layer', return_sequences= True, kernel_initializer= tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed= 123)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.LSTM(100, name='first_layer', return_sequences= True ),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.LSTM(100, name='first_layer', return_sequences= True ),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(68)
    ],
    name="generator",
)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(30,68), name="generator_input"),
        layers.LSTM(100, name='first_layer', kernel_initializer= tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed= 123)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(1),
    ],
    name="discriminator",
)

class GAN(keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator     

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        random_latent_vectors = tf.random.normal(shape=(batch_size, 30,68), seed=123)

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, tf.cast(real_images, dtype = tf.float32)], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((tf.shape(real_images)[0], 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.005 * tf.random.uniform(tf.shape(labels), seed= 123 )

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, 30,68), seed=1234)

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}

    
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img = 3):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, 30,68 ), seed= 123)
        generated_images = self.model.generator(random_latent_vectors)
        generated_images.numpy()
        generator_results.append(generated_images.numpy())

        discriminator_score = self.model.discriminator(generated_images)
        discriminator_score.numpy()
        discriminator_results.append(discriminator_score.numpy())


G_weights = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: generator_weights.append(generator.trainable_weights))

D_weights = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: generator_weights.append(generator.trainable_weights))

gan = GAN(discriminator=discriminator, generator=generator)

gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, )

dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)

history = gan.fit(
    dataset, 
    epochs=epochs, 
    callbacks=[GANMonitor(num_img=10)
               ,G_weights
               ,D_weights
               ,callbacks
               ]
)
