"""
Import packages & dataset

"""

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
