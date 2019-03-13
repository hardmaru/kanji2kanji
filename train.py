# Train all the required models

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # make TF only use the first GPU.

# imports

from kanji2kanji import *

# loads the data we need
unicode_label, test_unicode_label, old_train, old_test, new_kanji = load_data()

# train VAE on Kuzushiji Kanji
train_vae(unicode_label, old_train, new_kanji, vae_name="vae_old", kuzushiji=True)

# train VAE on Modern Kanji
train_vae(unicode_label, old_train, new_kanji, vae_name="vae_new", kuzushiji=False)

# train domain transfer MDN network:
train_latent2latent(unicode_label, old_train, new_kanji)

# train sketch-rnn model
train_kanji_rnn()

# run trained models on test set:

# loads the test data we need
kanji_set, new_kanji = load_kanji_rnn_test_data()

# loads trained models

reset_graph()

# loads domain transfer model
translate_model = Latent2Latent(batch_size=1, gpu_mode=False, is_training=False)
translate_model.load_checkpoint("image2image")

# loads VAE trained on Kuzushiji
vae_old = ConvVAE(z_size=Z_SIZE, batch_size=1, gpu_mode=False, is_training=False, reuse=True)
vae_old.load_checkpoint("vae_old")

# loads VAE trained on modern kanji
vae_new = ConvVAE(z_size=Z_SIZE, batch_size=1, gpu_mode=False, is_training=False, reuse=True)
vae_new.load_checkpoint("vae_new")

# load sketch-rnn model
hps_train, hps_test, hps_sample = get_default_hparams()
sample_model = MDNRNN(hps_sample, gpu_mode=False, layer_norm=True)
sample_model.load_checkpoint("kanji_rnn")

# sample results and put it in result dir:
for i in range(kanji_set.num_batches):
  dump_result(i, kanji_set, new_kanji, translate_model, vae_old, vae_new, sample_model, display_mode=False)
