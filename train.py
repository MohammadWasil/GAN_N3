import random
import numpy as np
import os
import time

from data_loading import generate_latent_points_fix, generate_real_samples_smoothed, generate_fake_samples, generate_latent_points

def Training(g_model, d_model, gan_model, X, y , latent_dim, args, n_epochs=10000, prefix="", epoch_start = 0):

    n_classes = args.NUMBER_OF_CLASSES
    n_batch = args.BATCH_SIZE

    # Only write csv header if first epoch
    if(epoch_start == 0):
        with open(os.path.join(args.LOG, f"{prefix}training.csv"), "w+") as f:
            f.write("epoch; d_loss_real; d_loss_fake; g_loss\n")

    bat_per_epo = int(X.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    # generate input and labels for progression tracking
    save_z_full, save_label_full = generate_latent_points_fix(latent_dim, n_classes=40)
    save_z_variety, _ = generate_latent_points_fix(latent_dim, n_classes=24) #small hack: 24 here to get 24 pictures, the actual class label comes later in this code
    
    if(epoch_start == 0):
        np.save(os.path.join(args.LOG, f"{prefix}save_z_full.npy"), save_z_full)
        np.save(os.path.join(args.LOG, f"{prefix}save_z_variety.npy"), save_z_variety)
    else:
        save_z_full = np.load(os.path.join(args.LOG, f"{prefix}save_z_full.npy"))
        save_z_variety = np.load(os.path.join(args.LOG, f"{prefix}save_z_variety.npy"))
    
    # manually enumerate epochs
    for i in range(epoch_start,n_epochs):
        start_t = time.time()
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples_smoothed(X, y, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)

            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, args)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)

            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch, args)

            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            if (j%50==0):
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # save the generator model
        if (i%5==0):
            print("Saving the model...")
            g_model.save(os.path.join(args.CHECKPOINTS,f'{prefix}cgan_generator_epoch_{i}.h5'))
            d_model.save(os.path.join(args.CHECKPOINTS,f'{prefix}cgan_discriminator_epoch_{i}.h5'))
            print("Saving visual progress...")
            plot_random_with_discr(d_model, g_model, i, latent_dim=100, save_pref=prefix)
        if(i%7==0):
            for i_label in range(n_classes):
                save_label_variety = np.array([i_label]*save_z_variety.shape[0])
                save_progress_variety(g_model, save_z_variety, save_label_variety, i, prefix = prefix, classname=f"{class_header[i_label]}")
        
        with open(os.path.join(args.LOG, f"{prefix}training.csv"), "a") as f:
            f.write(f"{i}; {d_loss1}; {d_loss2}; {g_loss}\n")
        end_t = time.time()
        
        print(f"epoch {i}, elapsed time: {data_loading.hms_string(end_t-start_t)}")
        if(n_classes == 40):
            save_progress_full_40(g_model, save_z_full, save_label_full, i, class_header, args, prefix = prefix)
        else:
            save_progress_full(g_model, save_z_full, save_label_full, i, class_header, args, prefix = prefix)